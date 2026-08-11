import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from datetime import datetime, timezone
import uuid
import weakref

import torch

from .prompting import (
    SmartTilePromptResolver,
    _apply_known_false_detection_guard,
    _apply_missing_local_caption_guard,
    _apply_plain_local_caption_guard,
    _canonical_surface_text,
    _excluded_detection_problem,
    _loads_tolerant,
    _object_candidate_label,
    _object_candidate_tokens,
    _object_text_tokens,
    _safe_view_context,
    _select_canonical_surface,
    _source_caption_problem,
    _surface_covers_tile,
    _surface_prompt_problem,
)
from .regions import (
    NINE_LOCATION_LABELS,
    flat_regions,
    measured_regions_text,
    uncovered_region_problem,
)


CACHE_SCHEMA_VERSION = 2
DEFAULT_ESRGAN_CACHE_MAX_GB = 2.0
DEFAULT_CACHE_FREE_RESERVE_GB = 2.0
# The text-prompt cache is bounded too: no cache in this tool may grow forever.
DEFAULT_PROMPT_CACHE_MAX_MB = 256.0
DEFAULT_PROMPT_CACHE_MAX_AGE_DAYS = 30.0


_TEMPLATE_PLACEHOLDER = re.compile(
    r"^\s*(?:<\s*(?:subject|object|identity|part|area)\s*>|"
    r"\[\s*(?:subject|object|identity|part|area)\s*\]|"
    r"\{\s*(?:subject|object|identity|part|area)\s*\}|"
    r"(?:subject|object|identity|part|area))\s*$",
    re.IGNORECASE,
)


def _is_template_placeholder(value):
    return bool(_TEMPLATE_PLACEHOLDER.fullmatch(str(value or "")))


def _analysis_image(image, max_side):
    """Downscale the vision-model input for whole-image analysis.

    A scene brief needs scene-level understanding, not gigapixels: feeding the
    full enlarged composite through the vision encoder costs VRAM and seconds
    on EVERY question asked about the image. 0 disables the cap.
    """
    limit = int(max_side)
    if limit <= 0:
        return image
    height, width = int(image.shape[1]), int(image.shape[2])
    longest = max(height, width)
    if longest <= limit:
        return image
    scale = limit / float(longest)
    return torch.nn.functional.interpolate(
        image.movedim(-1, 1),
        size=(max(1, round(height * scale)), max(1, round(width * scale))),
        mode="area",
    ).movedim(1, -1).clamp(0.0, 1.0)


def _require_single_image(image, purpose):
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError(f"{purpose} expected one ComfyUI IMAGE tensor.")
    if int(image.shape[0]) != 1:
        raise ValueError(
            f"{purpose} accepts one image at a time. Split image batches before "
            "Smart Upscaler so cached prompts and scene context cannot mix images."
        )


def _caption_generation_policy(value):
    """Keep legacy deterministic labels in the Prompt Director cache namespace."""
    if str(value) == "Varied wording":
        return "Varied wording"
    return "Managed by Prompt Director (recommended)"


def _known_false_detections(prompt_system):
    if not isinstance(prompt_system, dict):
        return ()
    return prompt_system.get("known_false_detections", ())


def _canonical_surface_fallback(reference):
    """Safest known identity for a tile, only when selection is unambiguous."""
    if isinstance(reference, dict):
        surface = _select_canonical_surface(
            reference, {}, str(reference.get("evidence_class", ""))
        )
        if surface is not None:
            return _canonical_surface_text(surface).strip()
    return ""


def _tile_context_fallback(reference):
    """What the brief already established for this tile's area, as plain content.

    When a tile's own caption cannot be recovered - roughly one tile in six on a
    dense 30-tile run - the alternative is a content-free phrase that gives a
    denoise sampler nothing to render. Everything used here was already selected
    FOR THIS TILE: the whole-image pass validated it, and delivery filtered it by
    measured location, so nothing new is invented.

    A MEASURED surface that covers the tile wins outright and is used alone. It
    is the only source here derived from pixels rather than from the model's own
    location list, and it describes the whole crop, so nothing else belongs
    beside it. Skipping this step is what put a city skyline in the East River:
    an open-water tile carried `water` at overlap 1.0 AND two objects whose part
    hints said "city skyline" and "parked boat", and the hints were used.

    Only then do advisory sources apply. Materials cannot take over a tile by
    design, and `part_in_this_tile` is the brief's own statement of what belongs
    here - but as this failure showed, a part hint can be a confident wrong noun,
    so it is the last resort rather than the first.
    """
    if not isinstance(reference, dict):
        return ""

    covering = [
        candidate
        for candidate in (reference.get("canonical_surfaces") or [])
        if isinstance(candidate, dict) and _surface_covers_tile(candidate)
    ]
    if covering:
        def overlap_of(candidate):
            try:
                return float(candidate.get("spatial_overlap", 0.0))
            except (TypeError, ValueError):
                return 0.0

        covering.sort(key=overlap_of, reverse=True)
        # A true tie is real ambiguity; fall through rather than pick one.
        if len(covering) == 1 or overlap_of(covering[0]) > overlap_of(covering[1]):
            best = covering[0]
            text = " ".join(_canonical_surface_text(best).split()).strip(" ,;.")
            if text:
                return text

    pieces = []

    def add(value):
        text = " ".join(str(value or "").split()).strip(" ,;.")
        if text and text.casefold() not in {item.casefold() for item in pieces}:
            pieces.append(text)

    for candidate in reference.get("canonical_objects") or []:
        if isinstance(candidate, dict):
            # The expected local part, never the whole subject.
            for part in str(candidate.get("part_in_this_tile", "")).split(";"):
                add(part)
    for candidate in reference.get("canonical_materials") or []:
        if isinstance(candidate, dict):
            add(candidate.get("target_prompt") or candidate.get("identity"))
    if not pieces:
        return ""
    return ", ".join(pieces)


_EXPECTED_OBJECT_MIN_OVERLAP = 0.12


def _object_identity_tokens(candidate):
    # Kept as a local compatibility name for the tile guard and subject merge.
    # The shared helper understands canonical target aliases and visible parts.
    return _object_candidate_tokens(candidate)


def _expected_object_candidates(tile_reference):
    try:
        reference = json.loads(str(tile_reference))
    except (TypeError, json.JSONDecodeError):
        return []
    if str(reference.get("evidence_class", "")) != "structured":
        return []
    candidates = reference.get("canonical_objects")
    if not isinstance(candidates, list):
        return []
    expected = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        identity = str(candidate.get("identity", "")).strip()
        candidate_id = str(candidate.get("id", "")).strip()
        if (
            not identity
            or not candidate_id
            or _is_template_placeholder(identity)
            or _is_template_placeholder(candidate_id)
            or not _object_identity_tokens(candidate)
        ):
            continue
        try:
            overlap = float(candidate.get("spatial_overlap", 0.0))
        except (TypeError, ValueError):
            overlap = 0.0
        if overlap >= _EXPECTED_OBJECT_MIN_OVERLAP:
            expected.append(candidate)
    return expected


def _expected_object_problem(caption, tile_reference):
    candidates = _expected_object_candidates(tile_reference)
    if not candidates:
        return ""
    payload = _loads_tolerant(caption)
    if not isinstance(payload, dict):
        return ""
    selected_id = str(payload.get("object_id", "")).strip().casefold()
    evidence = " ".join(
        str(payload.get(key, ""))
        for key in (
            "local_caption",
            "dominant_region",
            "visible_boundaries",
            "object_prompt",
            "local_features",
            "corrections_applied",
            "target_prompt",
        )
    )
    evidence_tokens = _object_text_tokens(evidence)
    for candidate in candidates:
        candidate_id = str(candidate.get("id", "")).strip()
        if selected_id and selected_id == candidate_id.casefold():
            continue
        tokens = _object_identity_tokens(candidate)
        identity = str(candidate.get("identity", "")).strip()
        identity_mentioned = bool(
            identity
            and re.search(
                rf"(?<!\w){re.escape(identity.casefold())}(?!\w)",
                evidence.casefold(),
            )
        )
        if identity_mentioned or tokens & evidence_tokens:
            # A positive mention confirms the candidate. An explicit negative
            # mention records that the exact tile was checked and rejected it;
            # the resolver strips absence clauses before diffusion prompting.
            continue
        label = _object_candidate_label(candidate)
        return (
            f'structured tile omitted location-matched object candidate "{label}" '
            f'(id "{candidate_id}"); explicitly confirm it from these pixels or state '
            "that it is not visible"
        )
    return ""


def _global_caption_problem(value, prompt_system, measured_regions=()):
    problem = _excluded_detection_problem(value, _known_false_detections(prompt_system))
    if problem:
        return f"whole-image brief contains {problem}"
    if not isinstance(prompt_system, dict) or not prompt_system.get("unified_instruction_ui"):
        return ""
    payload = _loads_tolerant(value)
    if payload is None:
        return "whole-image brief is not valid JSON"
    surface_map = payload.get("surface_map")
    if not isinstance(surface_map, list):
        return "whole-image brief is missing the surface_map array"
    # Materials are not measurable like flat regions, but the model must at
    # least ANSWER the question - a missing map means it skipped it, and every
    # patterned surface (rug, brick, knit, foliage) loses its shared wording.
    if not isinstance(payload.get("material_map"), list):
        return "whole-image brief is missing the material_map array"
    # Pixel measurement found these large smooth regions; the brief must name each
    # one, or a flat water/sky tile is left with no identity and misreads.
    coverage_problem = uncovered_region_problem(surface_map, measured_regions)
    if coverage_problem:
        return f"whole-image {coverage_problem}"
    # surface_map and object_map may both be empty. When present, each entry needs the
    # fields a tile reuses for a canonical continuous surface or spanning object. An
    # object may have an empty locations list (candidates are offered everywhere and
    # confirmed from pixels); a surface's delivery is location-based, so it may not.
    required_fields = {"id", "locations", "identity", "target_prompt"}
    for group_name, group in (
        ("surface_map", surface_map),
        ("material_map", payload.get("material_map")),
        ("object_map", payload.get("object_map")),
    ):
        if group is None:
            continue
        if not isinstance(group, list):
            return f"whole-image brief {group_name} must be an array"
        for entry in group:
            if not isinstance(entry, dict) or not required_fields.issubset(entry):
                return f"whole-image {group_name} contains an invalid entry"
            locations = entry.get("locations")
            if not isinstance(locations, list) or (
                group_name in ("surface_map", "material_map") and not locations
            ):
                return f"whole-image {group_name} contains invalid locations"
            if any(
                not str(entry.get(field, "")).strip()
                for field in ("id", "identity", "target_prompt")
            ):
                return f"whole-image {group_name} contains an incomplete entry"
            if any(
                _is_template_placeholder(entry.get(field, ""))
                for field in ("id", "identity", "target_prompt")
            ):
                return f"whole-image {group_name} contains a template placeholder"
            parts = entry.get("parts")
            if parts is not None:
                if group_name != "object_map" or not isinstance(parts, dict) or not parts:
                    return f"whole-image {group_name} contains invalid parts"
                location_names = {
                    str(location).strip().casefold() for location in locations
                }
                if any(
                    not str(label).strip()
                    or not str(part).strip()
                    or _is_template_placeholder(label)
                    or _is_template_placeholder(part)
                    or str(label).strip().casefold() not in location_names
                    for label, part in parts.items()
                ):
                    return f"whole-image {group_name} contains invalid parts"
    # A surface phrase is stamped onto isolated tiles, so it must name a real
    # material and stay object-free ("reflections of surrounding buildings" would
    # paint buildings into every water tile). Style/time-of-day transforms may
    # legitimately put requested lighting and sky words into surface targets.
    allow_atmosphere = (
        str(prompt_system.get("operation_mode", "")) == "style_transform"
    )
    for map_name in ("surface_map", "material_map"):
        for entry in payload.get(map_name, []):
            entry_problem = _surface_prompt_problem(entry, allow_atmosphere)
            if entry_problem:
                return f"whole-image {map_name} {entry_problem}"
    return ""


def _sanitize_surface_map(value, allow_atmosphere=False):
    """Deterministically repair surface entries that failed prompt safety.

    When the model keeps describing a surface with process words or nearby objects
    even after a retry, fall back to the entry's clean identity as its canonical
    phrase; if even the identity is unsafe, drop the entry so the measured-region
    recovery can re-ask a focused question instead.
    """
    payload = _loads_tolerant(value)
    if payload is None:
        return value
    changed = False
    # Materials share the surface safety contract (no process words, no nearby
    # objects), so unsafe material wording gets the same identity fallback.
    for map_name in ("surface_map", "material_map"):
        group = payload.get(map_name)
        if not isinstance(group, list):
            continue
        cleaned = []
        for entry in group:
            if not isinstance(entry, dict) or not _surface_prompt_problem(
                entry, allow_atmosphere
            ):
                cleaned.append(entry)
                continue
            identity_only = dict(entry)
            identity_only["target_prompt"] = str(entry.get("identity", "")).strip()
            if identity_only["target_prompt"] and not _surface_prompt_problem(
                identity_only, allow_atmosphere
            ):
                cleaned.append(identity_only)
        payload[map_name] = cleaned
        changed = True
    if not changed:
        return value
    return json.dumps(payload, ensure_ascii=False)


def _repair_map_entries(value):
    """Deterministically salvage malformed surface/object entries.

    A model that returns an entry with a missing id, a string instead of a
    location list, or no target_prompt must never kill the run: fill what can
    be filled from the entry itself and drop only what is truly hopeless. The
    focused fallback questions refill anything that gets dropped.
    """
    payload = _loads_tolerant(value)
    if payload is None:
        return value
    for group_name in ("surface_map", "material_map", "object_map"):
        group = payload.get(group_name)
        if not isinstance(group, list):
            continue
        repaired = []
        for index, entry in enumerate(group, start=1):
            if not isinstance(entry, dict):
                continue
            identity = str(entry.get("identity", "")).strip()
            entry_id = str(entry.get("id", "")).strip()
            target_value = str(entry.get("target_prompt", "")).strip()
            if any(
                _is_template_placeholder(value)
                for value in (identity, entry_id, target_value)
            ):
                continue
            if not identity and not entry_id:
                continue
            if not identity:
                identity = entry_id.replace("_", " ")
            if not entry_id:
                entry_id = f"{group_name.split('_')[0]}_{index}"
            locations = entry.get("locations")
            if isinstance(locations, str):
                locations = [locations]
            if not isinstance(locations, list):
                locations = []
            locations = [str(item)[:40] for item in locations[:9]]
            if group_name in ("surface_map", "material_map") and not locations:
                # Delivery to tiles is location-based; an unplaced surface or
                # material can never reach a tile, so the entry is hopeless.
                continue
            target_prompt = target_value or identity
            fixed = {
                "id": entry_id,
                "locations": locations,
                "identity": identity,
                "target_prompt": target_prompt,
            }
            if isinstance(entry.get("parts"), dict):
                location_names = {
                    str(location).strip().casefold() for location in locations
                }
                cleaned_parts = {
                    str(label).strip(): str(part).strip()
                    for label, part in entry["parts"].items()
                    if str(label).strip()
                    and str(part).strip()
                    and not _is_template_placeholder(label)
                    and not _is_template_placeholder(part)
                    and str(label).strip().casefold() in location_names
                }
                if cleaned_parts:
                    fixed["parts"] = cleaned_parts
            if str(entry.get("source_appearance", "")).strip():
                fixed["source_appearance"] = str(entry["source_appearance"]).strip()
            repaired.append(fixed)
        payload[group_name] = repaired
    return json.dumps(payload, ensure_ascii=False)


def _apply_global_brief_guard(value):
    """Ensure the surface/object arrays exist so downstream tile filtering is safe."""
    payload = _loads_tolerant(value)
    if payload is None:
        return value
    if not isinstance(payload.get("surface_map"), list):
        payload["surface_map"] = []
    if not isinstance(payload.get("material_map"), list):
        payload["material_map"] = []
    if not isinstance(payload.get("object_map"), list):
        payload["object_map"] = []
    return json.dumps(payload, ensure_ascii=False)


def _cache_root():
    override = os.environ.get("SMART_UPSCALER_CACHE_DIR")
    if override:
        root = Path(override)
    else:
        try:
            import folder_paths

            root = Path(folder_paths.get_user_directory()) / "smart_upscaler_cache"
        except ImportError:
            root = Path(tempfile.gettempdir()) / "smart_upscaler_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _positive_gigabytes_from_env(name, default):
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    return max(0.0, value)


def _remove_file_quietly(path):
    try:
        Path(path).unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _prepare_esrgan_cache_space(directory, required_bytes):
    """Bound the optional tensor cache and preserve free space for ComfyUI outputs."""
    directory = Path(directory)
    for temporary in directory.glob(".*.tmp"):
        _remove_file_quietly(temporary)
    for temporary in directory.glob("*.tmp"):
        _remove_file_quietly(temporary)

    max_bytes = int(
        _positive_gigabytes_from_env(
            "SMART_UPSCALER_ESRGAN_CACHE_MAX_GB", DEFAULT_ESRGAN_CACHE_MAX_GB
        )
        * 1024**3
    )
    reserve_bytes = int(
        _positive_gigabytes_from_env(
            "SMART_UPSCALER_CACHE_FREE_RESERVE_GB", DEFAULT_CACHE_FREE_RESERVE_GB
        )
        * 1024**3
    )
    entries = []
    total_bytes = 0
    for path in directory.glob("*.pt"):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((stat.st_mtime, path, stat.st_size))
        total_bytes += stat.st_size

    # A zero max disables persistent ESRGAN writes without disabling reads.
    if max_bytes <= 0 or required_bytes > max_bytes:
        return False, "entry exceeds the configured preprocessing-cache limit"

    entries.sort(key=lambda item: item[0])
    while entries:
        try:
            free_bytes = shutil.disk_usage(directory).free
        except OSError:
            free_bytes = 0
        if (
            total_bytes + required_bytes <= max_bytes
            and free_bytes >= required_bytes + reserve_bytes
        ):
            break
        _, path, size = entries.pop(0)
        if _remove_file_quietly(path):
            total_bytes = max(0, total_bytes - size)

    try:
        free_bytes = shutil.disk_usage(directory).free
    except OSError as exc:
        return False, f"storage check failed: {exc}"
    if total_bytes + required_bytes > max_bytes:
        return False, "preprocessing cache is at its configured size limit"
    if free_bytes < required_bytes + reserve_bytes:
        return False, "insufficient disk space after cache pruning"
    return True, ""


def _tensor_fingerprint(tensor):
    digest = hashlib.sha256()
    _update_digest_with_tensor(digest, tensor)
    return digest.hexdigest()


_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_MODEL_FINGERPRINT_CACHE = weakref.WeakKeyDictionary()
_MODEL_FINGERPRINT_LOCK = threading.RLock()


def _update_digest_with_tensor(digest, tensor):
    """Hash tensor storage in bounded chunks without a full ``tobytes`` copy."""
    value = tensor.detach().contiguous().cpu()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    if not value.numel():
        return
    byte_view = memoryview(value.view(torch.uint8).numpy()).cast("B")
    for offset in range(0, len(byte_view), _HASH_CHUNK_BYTES):
        digest.update(byte_view[offset : offset + _HASH_CHUNK_BYTES])


def _model_fingerprint(upscale_model):
    model = getattr(upscale_model, "model", upscale_model)
    if model is not None:
        try:
            with _MODEL_FINGERPRINT_LOCK:
                cached = _MODEL_FINGERPRINT_CACHE.get(model)
            if cached:
                return cached
        except TypeError:
            pass

    digest = hashlib.sha256()
    digest.update(type(upscale_model).__module__.encode("utf-8"))
    digest.update(type(upscale_model).__qualname__.encode("utf-8"))
    digest.update(type(model).__module__.encode("utf-8"))
    digest.update(type(model).__qualname__.encode("utf-8"))
    digest.update(str(getattr(upscale_model, "scale", 1.0)).encode("ascii"))

    try:
        state = model.state_dict()
        for name in sorted(state):
            value = state[name]
            if not isinstance(value, torch.Tensor):
                continue
            digest.update(name.encode("utf-8"))
            _update_digest_with_tensor(digest, value)
    except Exception:
        digest.update(repr(model).encode("utf-8", errors="replace"))
    fingerprint = digest.hexdigest()
    if model is not None:
        try:
            with _MODEL_FINGERPRINT_LOCK:
                _MODEL_FINGERPRINT_CACHE[model] = fingerprint
        except TypeError:
            pass
    return fingerprint


def esrgan_cache_key(image, upscale_model, settings, cache_tag):
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "kind": "esrgan_tiles",
        "image": _tensor_fingerprint(image),
        "model": _model_fingerprint(upscale_model),
        "cache_tag": str(cache_tag).strip(),
        "settings": settings,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_esrgan_tiles(cache_key, expected_shape):
    path = _cache_root() / "esrgan" / f"{cache_key}.pt"
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        tiles = payload.get("tiles") if isinstance(payload, dict) else None
        if not isinstance(tiles, torch.Tensor) or tuple(tiles.shape) != tuple(expected_shape):
            _remove_file_quietly(path)
            return None
        # Mark successful use so bounded pruning behaves like a recent-use cache.
        try:
            path.touch()
        except OSError:
            pass
        return tiles
    except Exception:
        _remove_file_quietly(path)
        return None


def save_esrgan_tiles(cache_key, tiles):
    directory = _cache_root() / "esrgan"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{cache_key}.pt"
    temporary = directory / f".{cache_key}.{uuid.uuid4().hex}.tmp"
    required_bytes = int(tiles.numel() * tiles.element_size()) + 8 * 1024**2
    ready, reason = _prepare_esrgan_cache_space(directory, required_bytes)
    if not ready:
        return None, reason
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tiles": tiles.detach().contiguous().cpu(),
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception as exc:
        _remove_file_quietly(temporary)
        return None, f"{type(exc).__name__}: {exc}"
    return path, ""


def _prompt_cache_key(image, instruction, key_context, cache_tag):
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "kind": "prompt",
        "image": _tensor_fingerprint(image),
        "instruction": str(instruction),
        "key_context": str(key_context),
        "cache_tag": str(cache_tag).strip(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_prompt(cache_key):
    path = _cache_root() / "prompts" / f"{cache_key}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema") != CACHE_SCHEMA_VERSION:
        return None
    value = payload.get("text")
    if isinstance(value, str):
        try:
            path.touch()
        except OSError:
            pass
    return value if isinstance(value, str) else None


def _prune_prompt_cache(directory):
    """Keep the prompt cache small and self-cleaning.

    Entries older than SMART_UPSCALER_PROMPT_CACHE_MAX_AGE_DAYS are removed, and
    the newest entries are kept within SMART_UPSCALER_PROMPT_CACHE_MAX_MB. Runs
    on every write, so the cache can never silently grow without bound.
    """
    max_bytes = int(
        _positive_gigabytes_from_env(
            "SMART_UPSCALER_PROMPT_CACHE_MAX_MB", DEFAULT_PROMPT_CACHE_MAX_MB
        )
        * 1024**2
    )
    max_age_seconds = (
        _positive_gigabytes_from_env(
            "SMART_UPSCALER_PROMPT_CACHE_MAX_AGE_DAYS", DEFAULT_PROMPT_CACHE_MAX_AGE_DAYS
        )
        * 86400.0
    )
    now = datetime.now(timezone.utc).timestamp()
    entries = []
    for path in Path(directory).glob("*.json"):
        try:
            stat = path.stat()
        except OSError:
            continue
        if max_age_seconds > 0 and now - stat.st_mtime > max_age_seconds:
            _remove_file_quietly(path)
            continue
        entries.append((stat.st_mtime, path, stat.st_size))
    entries.sort(reverse=True)
    total = 0
    for modified, path, size in entries:
        total += size
        if max_bytes > 0 and total > max_bytes:
            _remove_file_quietly(path)


def _write_prompt(cache_key, text, cache_tag):
    directory = _cache_root() / "prompts"
    path = directory / f"{cache_key}.json"
    temporary = directory / f".{cache_key}.{uuid.uuid4().hex}.tmp"
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cache_tag": str(cache_tag).strip(),
        "text": str(text),
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
        _prune_prompt_cache(directory)
    except Exception as exc:
        _remove_file_quietly(temporary)
        return None, f"{type(exc).__name__}: {exc}"
    return path, ""


def _whole_image_context_review(prompt_text, prompt_system):
    if not isinstance(prompt_system, dict) or not prompt_system.get("unified_instruction_ui"):
        return str(prompt_text)
    blueprint = str(prompt_system.get("prompt_blueprint", "")).strip()
    return (
        f"{blueprint}\n\n"
        "GENERATED MASTER SCENE PROMPT (SPATIALLY FILTERED FOR EACH TILE):\n"
        f"{str(prompt_text)}"
    )


class SmartCachedTextGenerate:
    CATEGORY = "Smart Upscaler/Prompting"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt_text", "cache_status", "review_text")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"forceInput": True, "lazy": True}),
                "image": ("IMAGE", {"forceInput": True}),
                "prompt": ("STRING", {"forceInput": True}),
                "max_length": (
                    "INT",
                    {"default": 1024, "min": 1, "max": 32768, "label": "Caption Length Limit"},
                ),
                "sampling_mode": (
                    [
                        "Managed by Prompt Director (recommended)",
                        "Consistent caption",
                        "Varied wording",
                    ],
                    {
                        "default": "Managed by Prompt Director (recommended)",
                        "advanced": True,
                        "label": "Caption Wording (Automatic)",
                        "tooltip": "Leave Managed. The Prompt Director owns semantic choices; the older Consistent and Varied values remain only for loading older workflows.",
                    },
                ),
                "thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "advanced": True,
                        "label": "Allow Model Reasoning (Advanced)",
                        "tooltip": "Leave off for a clean caption-only response.",
                    },
                ),
                "use_default_template": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "advanced": True,
                        "label": "Use Caption Model's Chat Format",
                        "tooltip": "Recommended ON for Qwen3-VL. Disable only when a replacement caption model explicitly expects raw unformatted text.",
                    },
                ),
                "cache_mode": (
                    ["read_write", "refresh", "bypass"],
                    {"default": "read_write", "advanced": True},
                ),
                "cache_tag": (
                    "STRING",
                    {
                        "default": "qwen3vl_4b_fp8_master_scene_v22_uniform_surfaces_v13",
                        "advanced": True,
                    },
                ),
            },
            "optional": {
                "key_context": ("STRING", {"default": "", "forceInput": True}),
                "prompt_system": ("SMART_PROMPT_SYSTEM", {"forceInput": True}),
                "analysis_max_side": (
                    "INT",
                    {
                        "default": 1344,
                        "min": 0,
                        "max": 8192,
                        "step": 32,
                        "advanced": True,
                        "label": "Analysis Image Size Cap (Advanced)",
                        "tooltip": "Longest side of the image sent to the vision model for the whole-image brief and its follow-up questions. Smaller = faster and less VRAM on every question; 1344 loses nothing for a scene brief. 0 sends the full image.",
                    },
                ),
                "vision_model_id": (
                    "STRING",
                    {
                        "default": "Qwen3-VL-4B-FP8",
                        "advanced": True,
                        "label": "Vision Model Cache ID",
                        "tooltip": "Cache identity for the connected caption model. Change this when replacing or updating that model; this prevents old captions from another model being reused.",
                    },
                ),
            },
        }

    @staticmethod
    def _context(
        max_length, sampling_mode, thinking, use_default_template, key_context,
        analysis_max_side=1344, vision_model_id="Qwen3-VL-4B-FP8",
    ):
        return json.dumps(
            {
                "max_length": int(max_length),
                "sampling_mode": _caption_generation_policy(sampling_mode),
                "thinking": bool(thinking),
                "use_default_template": bool(use_default_template),
                "key_context": str(key_context),
                "analysis_max_side": int(analysis_max_side),
                "vision_model_id": str(vision_model_id).strip(),
            },
            sort_keys=True,
        )

    @staticmethod
    def _measured_augmentation(image, prompt, prompt_system):
        """Append the deterministic flat-region measurement to the whole-image ask.

        The measured regions come from the pixels alone, so the vision model can be
        required to identify each one instead of being trusted to volunteer them.
        Only the unified contract understands surface_map, so legacy prompts pass
        through untouched.
        """
        if not isinstance(prompt_system, dict) or not prompt_system.get(
            "unified_instruction_ui"
        ):
            return str(prompt), []
        measured = flat_regions(image)
        block = measured_regions_text(measured)
        if not block:
            return str(prompt), []
        return f"{prompt}\n\n{block}", measured

    @staticmethod
    def _plain_identity_phrase(value):
        """Reduce a focused-question answer to one short identity phrase."""
        text = str(value or "").strip()
        while text.endswith("<end_of_turn>"):
            text = text[: -len("<end_of_turn>")].rstrip()
        text = text.splitlines()[0].strip() if text else ""
        text = re.sub(
            r"^\s*(?:the\s+region\s+is|this\s+region\s+is|this\s+is|it\s+is|it's)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = text.strip().strip("\"'`").rstrip(" .")
        if (
            not text
            or text.startswith(("{", "["))
            or _is_template_placeholder(text)
            or re.search(r"<\s*(?:subject|object|identity|part|area)\s*>", text, re.I)
        ):
            return ""
        words = text.split()
        if len(words) > 8:
            return ""
        return " ".join(words)

    def _identify_missed_regions(self, text, measured, prompt_system, run_caption):
        payload = _loads_tolerant(text)
        if payload is None:
            return text
        surface_map = payload.get("surface_map")
        if not isinstance(surface_map, list):
            surface_map = []
        for index, region in enumerate(measured, start=1):
            if not uncovered_region_problem(surface_map, [region]):
                continue
            place = " and ".join(region.get("labels", []))
            question = (
                "Look at the complete image. Direct pixel measurement found one large smooth "
                f"{region.get('color_name', '')} area covering the {place} of the image, "
                f"about {round(float(region.get('area_fraction', 0)) * 100)}% of the frame. "
                "In 2 to 6 words, what is this smooth thing actually part of in this scene? "
                "Infer its identity from the complete image, without choosing from examples or "
                "copying words from this question. Answer only with the literal identifying phrase."
            )
            identity = self._plain_identity_phrase(run_caption(question, response_limit=48))
            if not identity or _excluded_detection_problem(
                identity, _known_false_detections(prompt_system)
            ):
                continue
            entry = {
                "id": f"measured_surface_{index}",
                "locations": list(region.get("labels", [])),
                "identity": identity,
                "target_prompt": identity,
            }
            # A focused answer must itself be a safe surface phrase.
            if _surface_prompt_problem(entry):
                continue
            surface_map.append(entry)
        payload["surface_map"] = surface_map
        if not isinstance(payload.get("object_map"), list):
            payload["object_map"] = []
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _parse_subject_answer(answer):
        """Extract (identity, locations, parts) from a free-form subject answer.

        Models rarely keep the exact requested format, so this parser accepts
        nearly anything: with or without the pipe, semicolon or comma separated,
        prose around the labels. It scans for the nine area labels (longest
        first, so "top center" is never miscounted as "center") and treats the
        text after each label, up to the next label, as that area's part.
        """
        answer = str(answer or "").strip()
        if not answer or answer.casefold().startswith("none"):
            return "", [], {}
        if "|" in answer:
            identity_text, location_text = answer.split("|", 1)
        else:
            identity_text, location_text = "", answer
        lowered = location_text.casefold()
        claimed = []
        found = []
        for label in sorted(NINE_LOCATION_LABELS, key=len, reverse=True):
            start = 0
            while True:
                position = lowered.find(label, start)
                if position == -1:
                    break
                span = (position, position + len(label))
                if not any(span[0] < c[1] and c[0] < span[1] for c in claimed):
                    claimed.append(span)
                    found.append((position, label))
                start = position + 1
        found.sort()
        if not identity_text and found:
            identity_text = location_text[: found[0][0]]
        identity = SmartCachedTextGenerate._plain_identity_phrase(identity_text)
        locations = []
        parts = {}
        for index, (position, label) in enumerate(found):
            if label not in locations:
                locations.append(label)
            end = found[index + 1][0] if index + 1 < len(found) else len(location_text)
            fragment = location_text[position + len(label) : end]
            fragment = fragment.strip(" \t\r\n:;,.-()")
            words = fragment.split()
            junk = (
                not fragment
                or not re.search(r"[a-zA-Z]", fragment)
                or len(words) > 8
                or words[0].casefold().rstrip(".") in ("etc", "and", "or", "however")
            )
            if not junk and label not in parts:
                parts[label] = fragment[:80]
        ordered = [label for label in NINE_LOCATION_LABELS if label in locations]
        return identity, ordered, parts

    def _identify_main_subject(self, text, prompt_system, run_caption):
        """One focused question when the brief lacks a properly segmented subject.

        A clear main subject that the brief skips or lists lazily with one
        location and no parts leaves tiles guessing at fragments. Ask once,
        definitively, and merge the segmented answer deterministically.
        """
        payload = _loads_tolerant(text)
        if payload is None:
            return text
        object_map = payload.get("object_map")
        if not isinstance(object_map, list):
            return text
        has_segmented_entry = False
        for entry in object_map:
            if not isinstance(entry, dict) or not isinstance(entry.get("parts"), dict):
                continue
            locations = {
                str(value).strip().casefold()
                for value in entry.get("locations", [])
                if str(value).strip()
            }
            if any(
                str(label).strip().casefold() in locations
                and str(value).strip()
                and not _is_template_placeholder(value)
                for label, value in entry["parts"].items()
            ):
                has_segmented_entry = True
                break
        if has_segmented_entry:
            return text
        question = (
            "Look at the complete image. Does it show one clear main subject? Do not "
            "choose from examples or copy a subject name from this question. If yes, "
            "write its literal name, then a vertical bar, then each allowed image area "
            "followed by a colon and the visible piece occupying that area. Separate "
            "areas with semicolons. Never repeat instruction words as answer values. "
            "using only these area names: top left, top center, top right, middle left, "
            "center, middle right, bottom left, bottom center, bottom right. Cover every "
            "area any part of it touches, and name it plainly and with certainty. "
            "If there is no single main subject, answer only: none"
        )
        answer = str(run_caption(question, response_limit=160)).strip()
        identity, locations, parts = self._parse_subject_answer(answer)
        if not locations and not answer.casefold().startswith("none"):
            # The model chattered instead of following the format. One strict
            # re-ask; if that also fails, proceed without parts rather than
            # blocking the run.
            strict = (
                "Answer with no explanation. First write the literal main-subject name, "
                "then a vertical bar. After it, list every image area it touches as an "
                "allowed area name, a colon, and the visible piece in that area; separate "
                "entries with semicolons. Allowed area names: "
                + ", ".join(NINE_LOCATION_LABELS)
                + ". Never output angle-bracket placeholders or repeat the words subject, "
                "area, or part as values. If no main subject exists, answer only: none"
            )
            answer = str(run_caption(strict, response_limit=160)).strip()
            identity, locations, parts = self._parse_subject_answer(answer)
        if not locations:
            return text
        if not identity or _excluded_detection_problem(
            identity, _known_false_detections(prompt_system)
        ):
            return text
        named_parts = {
            label: part
            for label, part in parts.items()
            if part and not _is_template_placeholder(part)
        }
        identity_tokens = _object_text_tokens(identity)
        matched_entry = None
        best_match = 0
        for entry in object_map:
            if not isinstance(entry, dict):
                continue
            existing_identity = str(entry.get("identity", "")).strip().casefold()
            existing_tokens = _object_identity_tokens(entry)
            score = len(identity_tokens & existing_tokens)
            if existing_identity == identity.casefold():
                score += 100
            if score > best_match:
                best_match = score
                matched_entry = entry

        if matched_entry is not None and best_match > 0:
            # Merge segmentation only into the same named subject. A focused
            # answer such as "person" must never turn the first unrelated map
            # entry (for example a bridge) into a person/bridge hybrid.
            existing = matched_entry.get("locations")
            existing = [str(v) for v in existing] if isinstance(existing, list) else []
            matched_entry["locations"] = existing + [
                label for label in locations if label not in existing
            ]
            if named_parts:
                matched_entry["parts"] = named_parts
            # Refine a broad class such as "building" when the focused whole-image
            # check supplied the literal subject identity ("wooden cabin"). The
            # canonical target remains untouched, preserving its richer wording.
            if identity_tokens and not _object_text_tokens(existing_identity):
                matched_entry["identity"] = identity
            if not str(matched_entry.get("target_prompt", "")).strip():
                matched_entry["target_prompt"] = identity
        else:
            used_ids = {
                str(entry.get("id", "")).casefold()
                for entry in object_map
                if isinstance(entry, dict)
            }
            entry_id = "main_subject"
            suffix = 2
            while entry_id.casefold() in used_ids:
                entry_id = f"main_subject_{suffix}"
                suffix += 1
            entry = {
                "id": entry_id,
                "locations": locations,
                "identity": identity,
                "target_prompt": identity,
            }
            if named_parts:
                entry["parts"] = named_parts
            object_map.append(entry)
        payload["object_map"] = object_map
        return json.dumps(payload, ensure_ascii=False)

    def check_lazy_status(
        self, clip, image, prompt, max_length, sampling_mode, thinking,
        use_default_template, cache_mode, cache_tag, key_context="", prompt_system=None,
        analysis_max_side=1344, vision_model_id="Qwen3-VL-4B-FP8",
    ):
        _require_single_image(image, "Whole-image captioning")
        if clip is not None:
            return []
        context = self._context(
            max_length, sampling_mode, thinking, use_default_template, key_context,
            analysis_max_side, vision_model_id,
        )
        if cache_mode == "read_write":
            prompt, measured = self._measured_augmentation(image, prompt, prompt_system)
            vision_image = _analysis_image(image, analysis_max_side)
            cache_key = _prompt_cache_key(vision_image, prompt, context, cache_tag)
            cached = _read_prompt(cache_key)
            if cached is not None and not _global_caption_problem(
                cached, prompt_system, measured
            ):
                return []
        return ["clip"]

    def generate(
        self, clip, image, prompt, max_length, sampling_mode, thinking,
        use_default_template, cache_mode, cache_tag, key_context="", prompt_system=None,
        analysis_max_side=1344, vision_model_id="Qwen3-VL-4B-FP8",
    ):
        _require_single_image(image, "Whole-image captioning")
        context = self._context(
            max_length, sampling_mode, thinking, use_default_template, key_context,
            analysis_max_side, vision_model_id,
        )
        prompt, measured = self._measured_augmentation(image, prompt, prompt_system)
        vision_image = _analysis_image(image, analysis_max_side)
        cache_key = _prompt_cache_key(vision_image, prompt, context, cache_tag)
        if cache_mode == "read_write":
            cached = _read_prompt(cache_key)
            if cached is not None and not _global_caption_problem(
                cached, prompt_system, measured
            ):
                return (
                    cached,
                    f"Prompt cache HIT | {cache_key[:12]}",
                    _whole_image_context_review(cached, prompt_system),
                )
        if clip is None:
            raise ValueError("The prompt model was not evaluated for a cache miss.")

        def run_caption(instruction, response_limit=None):
            tokens = clip.tokenize(
                instruction,
                image=vision_image,
                skip_template=not use_default_template,
                min_length=1,
                thinking=thinking,
            )
            generated_ids = clip.generate(
                tokens,
                do_sample=sampling_mode in ("on", "Varied wording"),
                max_length=int(response_limit or max_length),
                temperature=0.7,
                top_k=64,
                top_p=0.95,
                min_p=0.05,
                repetition_penalty=1.05,
                presence_penalty=0.0,
                seed=0,
            )
            return str(clip.decode(generated_ids))

        text = run_caption(prompt)
        problem = _global_caption_problem(text, prompt_system, measured)
        retried = False
        if problem:
            false_terms = ", ".join(_known_false_detections(prompt_system))
            repair_prompt = f"{prompt}\n\nThe previous brief was invalid: {problem}."
            if false_terms:
                repair_prompt += (
                    f" The user confirms these detections are false: {false_terms}. Return the "
                    "requested JSON without any false-detection term."
                )
            else:
                repair_prompt += " Return the corrected complete JSON brief."
            text = run_caption(repair_prompt)
            retried = True
            problem = _global_caption_problem(text, prompt_system, measured)
        if problem and "false detection" not in problem:
            text = _apply_global_brief_guard(text)
            problem = _global_caption_problem(text, prompt_system, measured)
        if problem and (
            "invalid entry" in problem
            or "incomplete entry" in problem
            or "invalid locations" in problem
            or "invalid parts" in problem
            or "template placeholder" in problem
        ):
            # Broken entry shapes are salvaged, never fatal.
            text = _repair_map_entries(text)
            problem = _global_caption_problem(text, prompt_system, measured)
            retried = True
        if problem and (
            "surface_map" in problem or "material_map" in problem
        ) and (
            "naming the real material" in problem
            or "only the surface itself" in problem
        ):
            text = _sanitize_surface_map(
                text,
                allow_atmosphere=(
                    isinstance(prompt_system, dict)
                    and str(prompt_system.get("operation_mode", "")) == "style_transform"
                ),
            )
            problem = _global_caption_problem(text, prompt_system, measured)
            retried = True
        unidentified_region = False
        if problem and "measured flat region" in problem:
            # The model skipped a region the pixels prove exists. Ask one focused
            # question per missed region — identity only, judged from the whole
            # image — and add the answer as a canonical surface deterministically.
            text = self._identify_missed_regions(
                text, measured, prompt_system, run_caption
            )
            problem = _global_caption_problem(text, prompt_system, measured)
            retried = True
        if problem and "measured flat region" in problem:
            # Even the focused question could not name this region. Proceed safely
            # rather than kill the run: tiles there use their own pixels plus
            # camera context. The brief is not cached, so the next run asks again.
            unidentified_region = True
            problem = ""
        if problem:
            raise ValueError(
                f"Whole-image task brief rejected after retry: {problem}. "
                "Unsafe scene context was not passed to tile prompting."
            )
        if isinstance(prompt_system, dict) and prompt_system.get("unified_instruction_ui"):
            enriched = self._identify_main_subject(text, prompt_system, run_caption)
            if enriched != text and not _global_caption_problem(
                enriched, prompt_system, measured
            ):
                text = enriched
        if unidentified_region:
            return (
                text,
                "Prompt cache BYPASS | a measured flat region stayed unidentified",
                _whole_image_context_review(text, prompt_system),
            )
        if cache_mode != "bypass":
            _, write_error = _write_prompt(cache_key, text, cache_tag)
            if write_error:
                return (
                    text,
                    f"Prompt cache WRITE SKIPPED | {write_error}",
                    _whole_image_context_review(text, prompt_system),
                )
            action = "RETRY WRITE" if retried else "WRITE"
            return (
                text,
                f"Prompt cache {action} | {cache_key[:12]}",
                _whole_image_context_review(text, prompt_system),
            )
        return text, "Prompt cache BYPASS", _whole_image_context_review(text, prompt_system)


class SmartCachedTilePromptGenerator:
    """Generate a deterministic local caption, cache it, and assemble the final tile prompt."""

    CATEGORY = "Smart Upscaler/Prompting"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "positive_prompt",
        "negative_prompt",
        "prompt_audit",
        "tile_reference",
        "cache_status",
    )
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"forceInput": True, "lazy": True}),
                "image": ("IMAGE", {"forceInput": True}),
                "instruction": ("STRING", {"forceInput": True}),
                "tile_reference": ("STRING", {"forceInput": True}),
                "prompt_system": ("SMART_PROMPT_SYSTEM", {"forceInput": True}),
                "caption_generation": (
                    [
                        "Managed by Prompt Director (recommended)",
                        "Consistent caption (recommended)",
                        "Varied wording",
                    ],
                    {
                        "default": "Managed by Prompt Director (recommended)",
                        "advanced": True,
                        "label": "Caption Wording (Automatic)",
                        "tooltip": "Leave Managed. Tile detail and context now come from the Prompt Director; older choices remain only for loading older workflows.",
                    },
                ),
                "cache_mode": (
                    ["read_write", "refresh", "bypass"],
                    {
                        "default": "read_write",
                        "advanced": True,
                        "tooltip": "Reuse saved captions, regenerate them, or skip the cache.",
                    },
                ),
                "cache_tag": (
                    "STRING",
                    {
                        "default": "qwen3vl_4b_fp8_local_target_v29_structured_detail_v13",
                        "advanced": True,
                        "tooltip": "Change this after changing the caption model or caption contract.",
                    },
                ),
                "negative_prompt_fallback": (
                    "STRING",
                    {
                        "default": "changed geometry, duplicated objects, artifacts, seams",
                        "multiline": True,
                        "advanced": True,
                    },
                ),
            },
            "optional": {
                "caption_max_side": (
                    "INT",
                    {
                        "default": 1344,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "advanced": True,
                        "label": "Caption Image Size Cap (Advanced)",
                        "tooltip": "Longest side of each real (unpadded) tile sent to the vision model. 1344 is the balanced 16 GB default; 0 sends the full tile, while 768-1024 saves more VRAM with a small fine-detail cost.",
                    },
                ),
                "vision_model_id": (
                    "STRING",
                    {
                        "default": "Qwen3-VL-4B-FP8",
                        "advanced": True,
                        "label": "Vision Model Cache ID",
                        "tooltip": "Cache identity for the connected caption model. Change this when replacing or updating that model; this prevents old tile captions from another model being reused.",
                    },
                ),
            },
        }

    @staticmethod
    def _uses_direct_user(prompt_system):
        return str(prompt_system.get("prompt_strategy", "task_directed")) == "direct_user"

    @staticmethod
    def _caption_token_budget(prompt_system):
        """Room for the whole JSON answer at the chosen Prompt Detail Level.

        The tile returns local_caption, dominant_region, visible_boundaries,
        local_features, corrections_applied AND target_prompt in one JSON object.
        A flat 256-token cap fit the short levels but cut Maximum off mid-word,
        so choosing more detail produced a WORSE prompt than choosing less -
        "The foreground shows a," reached the sampler verbatim.
        """
        detail = ""
        if isinstance(prompt_system, dict):
            detail = str(prompt_system.get("caption_detail", ""))
        return {
            "Simple": 256,
            "Adaptive by Tile (recommended)": 320,
            "Complex": 448,
            "Detailed": 512,
            "Maximum (every visible item)": 768,
        }.get(detail, 320)

    @classmethod
    def _key_context(
        cls,
        tile_reference,
        caption_generation,
        caption_max_side=1344,
        vision_model_id="Qwen3-VL-4B-FP8",
        prompt_system=None,
    ):
        return json.dumps(
            {
                "tile_reference": str(tile_reference),
                "caption_generation": _caption_generation_policy(caption_generation),
                "thinking": False,
                "template": True,
                "max_length": cls._caption_token_budget(prompt_system),
                "caption_max_side": int(caption_max_side),
                "vision_model_id": str(vision_model_id).strip(),
            },
            sort_keys=True,
        )

    @staticmethod
    def _deterministic_uniform_caption(tile_reference):
        """Skip the vision model for uniform tiles with a canonical surface.

        The resolver replaces such a tile's caption with the canonical surface
        text regardless of what the model says, so generating a caption for a
        water/sky/bokeh tile is pure wasted seconds. Build the equivalent
        response deterministically instead.
        """
        try:
            reference = json.loads(str(tile_reference))
        except json.JSONDecodeError:
            return None
        if str(reference.get("evidence_class", "")) != "uniform":
            return None
        surface = _select_canonical_surface(reference, {}, "uniform")
        if surface is None:
            return None
        identity = str(surface.get("identity", "")).strip()
        surface_id = str(surface.get("id", "")).strip()
        target = str(surface.get("target_prompt", "")).strip() or identity
        if not identity or not surface_id or not target:
            return None
        return json.dumps(
            {
                "local_caption": identity,
                "dominant_region": identity,
                "visible_boundaries": "",
                "surface_id": surface_id,
                "surface_prompt": target,
                "object_id": "",
                "object_prompt": "",
                "local_features": "",
                "corrections_applied": "",
                "target_prompt": target,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _evidence_class(tile_reference):
        try:
            reference = json.loads(str(tile_reference))
        except json.JSONDecodeError:
            return "structured"
        return str(reference.get("evidence_class", "structured"))

    @staticmethod
    def _visual_complexity(tile_reference):
        try:
            reference = json.loads(str(tile_reference))
        except json.JSONDecodeError:
            return ""
        return str(reference.get("visual_complexity", ""))

    @staticmethod
    def _requires_target(prompt_system):
        return str(prompt_system.get("composition_mode", "")) == "task_directed"

    def _caption_problem(self, caption, tile_reference, prompt_system):
        problem = _source_caption_problem(
            caption,
            self._evidence_class(tile_reference),
            self._requires_target(prompt_system),
            _known_false_detections(prompt_system),
            self._visual_complexity(tile_reference),
            str(prompt_system.get("task_preset", "")) == "Google Image Enhance",
        )
        if problem:
            return problem
        return _expected_object_problem(caption, tile_reference)

    @staticmethod
    def _run_caption(clip, image, prompt, caption_generation, max_length=256):
        tokens = clip.tokenize(
            prompt,
            image=image,
            skip_template=False,
            min_length=1,
            thinking=False,
        )
        generated_ids = clip.generate(
            tokens,
            do_sample=str(caption_generation) == "Varied wording",
            max_length=max_length,
            temperature=0.7,
            top_k=64,
            top_p=0.95,
            min_p=0.05,
            repetition_penalty=1.05,
            presence_penalty=0.0,
            seed=0,
        )
        return str(clip.decode(generated_ids))

    def check_lazy_status(
        self,
        clip,
        image,
        instruction,
        tile_reference,
        prompt_system,
        caption_generation,
        cache_mode,
        cache_tag,
        negative_prompt_fallback,
        caption_max_side=1344,
        vision_model_id="Qwen3-VL-4B-FP8",
    ):
        _require_single_image(image, "Tile captioning")
        if self._uses_direct_user(prompt_system) or clip is not None:
            return []
        deterministic = self._deterministic_uniform_caption(tile_reference)
        if deterministic is not None and not self._caption_problem(
            deterministic, tile_reference, prompt_system
        ):
            return []
        if cache_mode == "read_write":
            caption_image = _analysis_image(image, caption_max_side)
            cache_key = _prompt_cache_key(
                caption_image,
                instruction,
                self._key_context(
                    tile_reference,
                    caption_generation,
                    caption_max_side,
                    vision_model_id,
                    # Must match generate() exactly: the token budget is part of
                    # the key, so omitting it here made every lazy check miss.
                    prompt_system,
                ),
                cache_tag,
            )
            cached = _read_prompt(cache_key)
            if cached is not None and not self._caption_problem(
                cached, tile_reference, prompt_system
            ):
                return []
        return ["clip"]

    def generate(
        self,
        clip,
        image,
        instruction,
        tile_reference,
        prompt_system,
        caption_generation,
        cache_mode,
        cache_tag,
        negative_prompt_fallback,
        caption_max_side=1344,
        vision_model_id="Qwen3-VL-4B-FP8",
    ):
        _require_single_image(image, "Tile captioning")
        resolver = SmartTilePromptResolver()
        if self._uses_direct_user(prompt_system):
            resolved = resolver.resolve(
                None,
                tile_reference,
                prompt_system,
                negative_prompt_fallback,
            )
            return (*resolved, "Tile caption BYPASSED | User Request only")

        deterministic = self._deterministic_uniform_caption(tile_reference)
        if deterministic is not None and not self._caption_problem(
            deterministic, tile_reference, prompt_system
        ):
            resolved = resolver.resolve(
                deterministic,
                tile_reference,
                prompt_system,
                negative_prompt_fallback,
            )
            return (
                *resolved,
                "Tile caption DETERMINISTIC | uniform canonical surface",
            )

        image = _analysis_image(image, caption_max_side)
        caption_tokens = self._caption_token_budget(prompt_system)
        context = self._key_context(
            tile_reference,
            caption_generation,
            caption_max_side,
            vision_model_id,
            prompt_system,
        )
        cache_key = _prompt_cache_key(image, instruction, context, cache_tag)
        caption = None
        cache_status = "Tile caption cache BYPASS"
        if cache_mode == "read_write":
            caption = _read_prompt(cache_key)
            if caption is not None and not self._caption_problem(
                caption, tile_reference, prompt_system
            ):
                cache_status = f"Tile caption cache HIT | {cache_key[:12]}"
            else:
                caption = None

        if caption is None:
            if clip is None:
                raise ValueError("The tile caption model was not evaluated for a cache miss.")
            caption = self._run_caption(
                clip, image, instruction, caption_generation, caption_tokens
            )
            problem = self._caption_problem(caption, tile_reference, prompt_system)
            retried = False
            false_detection_guarded = False
            schema_guarded = False
            plain_recovery_guarded = False
            conservative_fallback_guarded = False
            expected_objects = _expected_object_candidates(tile_reference)
            expected_object_rule = ""
            if expected_objects:
                hypotheses = "; ".join(
                    f'id "{candidate.get("id", "")}": {candidate.get("identity", "")}'
                    for candidate in expected_objects
                )
                expected_object_rule = (
                    " Whole-image location mapping strongly overlaps this tile with these "
                    f"object hypotheses: {hypotheses}. Reinspect each hypothesis against the "
                    "exact pixels. If visible, name the object and return its id/object prompt. "
                    "If absent, explicitly state that the named candidate is not visible; never "
                    "silently omit it."
                )

            def run_plain_recovery():
                try:
                    reference = json.loads(str(tile_reference))
                except json.JSONDecodeError:
                    reference = {}
                view_context = _safe_view_context(reference.get("view_context", ""))
                surface_prompt = _canonical_surface_fallback(reference)
                view_rule = (
                    f" If the pixels support it, retain this camera description: {view_context}."
                    if view_context
                    else ""
                )
                false_terms = ", ".join(_known_false_detections(prompt_system))
                false_rule = (
                    f" Omit these user-confirmed false detections: {false_terms}."
                    if false_terms
                    else ""
                )
                context_match = re.search(
                    r"(?:WHOLE-IMAGE TASK BRIEF|WHOLE-IMAGE REFERENCE FILTERED FOR THIS TILE):"
                    r"\s*(.*?)\nLocal evidence check:",
                    str(instruction),
                    flags=re.DOTALL,
                )
                context_hint = ""
                if context_match:
                    compact_context = " ".join(context_match.group(1).split())[:600]
                    if compact_context:
                        context_hint = (
                            " Whole-image context may help name visible pixels but does not prove "
                            f"local presence: {compact_context}."
                        )
                evidence_class = self._evidence_class(tile_reference)
                evidence_rule = (
                    " Deterministic pixel analysis confirms structured local content. Include the "
                    "visible foreground structures, surfaces, and boundaries; do not describe only "
                    "the background."
                    if evidence_class == "structured"
                    else ""
                )
                region_rule = (
                    " The whole-image analysis spatially matches this exact tile to this continuous "
                    f"surface: {surface_prompt}. Use that identity only if the pixels agree; "
                    "do not include any other scene objects."
                    if surface_prompt
                    else ""
                )
                recovery_instruction = (
                    "Inspect only the pixels in this exact image tile. Return one short factual "
                    "sentence describing the visible local source content, material or continuous "
                    "region. Do not return JSON, headings, reasoning, alternatives, or instructions. "
                    "Do not call the tile empty, a placeholder, or a background. Do not infer an "
                    "object from the complete scene unless its pixels are visible here."
                    f"{evidence_rule}{expected_object_rule}{region_rule}"
                    f"{context_hint}{view_rule}{false_rule}"
                )
                recovered = self._run_caption(
                    clip,
                    image,
                    recovery_instruction,
                    "Consistent caption",
                    max_length=96,
                )
                return _apply_plain_local_caption_guard(recovered)

            def build_conservative_fallback(recovery_reason):
                try:
                    reference = json.loads(str(tile_reference))
                except json.JSONDecodeError:
                    reference = {}
                evidence_class = self._evidence_class(tile_reference)
                view_context = _safe_view_context(reference.get("view_context", "")).rstrip(" .")
                surface_prompt = _canonical_surface_fallback(reference).rstrip(" .")
                if evidence_class == "uniform":
                    local = surface_prompt or "the locally visible uniform source surface"
                elif evidence_class == "sparse":
                    local = (
                        "the dominant source region and only the partial boundaries visible "
                        "in this exact tile"
                    )
                elif (
                    isinstance(prompt_system, dict)
                    and str(prompt_system.get("prompt_format", "")) == "description"
                ):
                    # A denoise sampler receives this text as the WHOLE prompt, so a
                    # sentence about "the source structures ... in this exact tile"
                    # gives it nothing to render and the tile drifts. Use whatever
                    # the brief already established for THIS tile's area; only fall
                    # back to content-free wording when there is nothing at all.
                    local = _tile_context_fallback(reference) or (
                        "a sharp, naturally detailed photograph"
                    )
                else:
                    local = _tile_context_fallback(reference) or (
                        "the source structures, surfaces, and boundaries visible in this exact tile"
                    )
                if view_context and evidence_class in ("uniform", "sparse"):
                    local = f"{view_context} of {local}"
                rejected_hypotheses = "; ".join(
                    f'{_object_candidate_label(candidate)} candidate is not confirmed '
                    "visible in this exact tile"
                    for candidate in expected_objects
                )
                return json.dumps(
                    {
                        "local_caption": local,
                        "dominant_region": local,
                        "visible_boundaries": "",
                        "surface_id": "",
                        "surface_prompt": "",
                        "local_features": "",
                        # This records that every whole-image hypothesis was checked
                        # but remains unconfirmed. It satisfies the semantic audit
                        # without copying an unproven object into the diffusion prompt.
                        "corrections_applied": rejected_hypotheses,
                        "target_prompt": local,
                        "recovery_reason": str(recovery_reason),
                    },
                    ensure_ascii=False,
                )

            if problem:
                repair_instruction = (
                    f"{instruction}\n\nThe previous response was invalid: {problem}. Inspect this exact "
                    "image tile again. It contains source pixels and is not a locator, mask, "
                    "placeholder, or empty background. Follow the whole-image task brief and target "
                    "task above. Analyze the pixels instead of repeating the output template. Return "
                    "complete JSON with local_caption, dominant_region, visible_boundaries, "
                    "surface_id, surface_prompt, object_id, object_prompt, local_features, "
                    f"corrections_applied, and target_prompt.{expected_object_rule}"
                )
                caption = self._run_caption(
                    clip, image, repair_instruction, caption_generation, caption_tokens
                )
                retried = True
                problem = self._caption_problem(caption, tile_reference, prompt_system)
            if problem == "missing local_caption":
                caption = _apply_missing_local_caption_guard(caption)
                schema_guarded = True
                problem = self._caption_problem(caption, tile_reference, prompt_system)
            if problem.startswith('caption contains user-confirmed false detection'):
                caption = _apply_known_false_detection_guard(
                    caption, _known_false_detections(prompt_system)
                )
                false_detection_guarded = True
                problem = self._caption_problem(caption, tile_reference, prompt_system)
            # Any remaining model-content or schema failure gets the same small,
            # reliable exact-tile recaption. Validation wording must never decide
            # whether an all-tiles run is allowed to continue.
            if problem:
                caption = run_plain_recovery()
                plain_recovery_guarded = True
                problem = self._caption_problem(caption, tile_reference, prompt_system)
                if problem.startswith('caption contains user-confirmed false detection'):
                    caption = _apply_known_false_detection_guard(
                        caption, _known_false_detections(prompt_system)
                    )
                    false_detection_guarded = True
                    problem = self._caption_problem(caption, tile_reference, prompt_system)
            if problem:
                recovery_reason = problem
                caption = build_conservative_fallback(recovery_reason)
                conservative_fallback_guarded = True
                problem = self._caption_problem(caption, tile_reference, prompt_system)
            if problem:
                try:
                    tile_id = json.loads(str(tile_reference)).get("tile_id", "unknown tile")
                except json.JSONDecodeError:
                    tile_id = "unknown tile"
                raise ValueError(
                    f"{tile_id} caption rejected after retry: {problem}. "
                    "The sampler was not given an unsafe prompt."
                )
            if cache_mode != "bypass":
                _, write_error = _write_prompt(cache_key, caption, cache_tag)
                guard_labels = []
                if schema_guarded:
                    guard_labels.append("SCHEMA")
                if plain_recovery_guarded:
                    guard_labels.append("PLAIN-RECOVERY")
                if conservative_fallback_guarded:
                    guard_labels.append("CONSERVATIVE-FALLBACK")
                if false_detection_guarded:
                    guard_labels.append("FALSE-DETECTION")
                if guard_labels:
                    action = f"RETRY {' + '.join(guard_labels)}-GUARD WRITE"
                else:
                    action = "RETRY WRITE" if retried else "WRITE"
                if write_error:
                    cache_status = f"Tile caption cache WRITE SKIPPED | {write_error}"
                else:
                    cache_status = f"Tile caption cache {action} | {cache_key[:12]}"
            elif (
                schema_guarded
                or plain_recovery_guarded
                or conservative_fallback_guarded
                or false_detection_guarded
            ):
                guards = []
                if schema_guarded:
                    guards.append("schema")
                if plain_recovery_guarded:
                    guards.append("plain-recovery")
                if conservative_fallback_guarded:
                    guards.append("conservative-fallback")
                if false_detection_guarded:
                    guards.append("false-detection")
                cache_status = (
                    "Tile caption cache BYPASS | deterministic " + " + ".join(guards) + " guard"
                )

        resolved = resolver.resolve(
            caption,
            tile_reference,
            prompt_system,
            negative_prompt_fallback,
        )
        return (*resolved, cache_status)
