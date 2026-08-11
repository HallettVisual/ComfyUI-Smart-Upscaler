import json
from pathlib import Path
import re

import torch
import torch.nn.functional as F

from .processing import SmartTileSeed
from .prompting import (
    _SURFACE_FAMILY_PATTERNS,
    _load_tile_metadata,
    _loads_tolerant,
    _false_detection_forms,
    _map_prompt_carries_nothing,
    _map_prompt_is_thin,
    _strip_reflected_objects,
    _surface_prompt_problem,
)
from .regions import location_axes, tile_mean_rgb, uniform_tile_components


SIMPLE_MASTER_INSTRUCTION = """Caption only the source pixels inside the exact tile. Whole-image context may clarify identity and continuity but never proves local presence. Do not add, move, or replace visible content."""


TASK_PRESET_CHOICES = (
    "Google Image Enhance",
    "Style Transfer",
    "Time of Day",
    "Upscale / Detailer",
)


TASK_PRESETS = {
    "Google Image Enhance": {
        "operation_mode": "detail_enhance",
        "direct_prompt": "Repair this exact Google Earth tile into a realistic photograph.",
        "instruction": (
            "Analyze the complete image as Google Earth or 3D map imagery, identify reconstruction "
            "and texture-warping artifacts, and direct each exact tile toward a coherent realistic "
            "photograph of the same intact scene."
        ),
        "global_rule": (
            "Treat melted, torn, stretched, duplicated, broken-looking, misaligned, or incomplete "
            "geometry and textures as photogrammetry reconstruction errors unless the complete "
            "scene clearly proves actual physical damage. Identify geographic context and large "
            "continuous regions when supportable. Never guess an environment type from color alone."
        ),
        "local_rule": (
            "The target prompt states what SHOULD be there — the intact, repaired result — never "
            "the current defect. Do not describe damage, wear, debris, or brokenness in the target "
            "prompt; describe the clean, coherent structure it should become. Treat isolated "
            "incoherent specks, thin lines, torn fragments, and ambiguous partial shapes as capture "
            "artifacts unless their pixels clearly form a real object."
        ),
    },
    "Style Transfer": {
        "operation_mode": "style_transform",
        "direct_prompt": "Convert this image to the requested style.",
        "instruction": (
            "Analyze the complete image and requested style, establish one consistent target "
            "appearance, and direct each exact tile to express that style using only its locally "
            "visible content."
        ),
        "global_rule": (
            "Separate source appearance from the requested target style. Define the target palette, "
            "lighting, texture, and rendering character consistently without changing scene content."
        ),
        "local_rule": (
            "The target prompt must express the requested style on the exact local content while "
            "preserving unrequested geometry, object identity, and boundaries."
        ),
    },
    "Time of Day": {
        "operation_mode": "style_transform",
        "direct_prompt": "Change this image to the requested time of day.",
        "instruction": (
            "Analyze the complete image and requested time of day, establish one consistent lighting "
            "and atmosphere plan, and direct each exact tile toward that result using only locally "
            "visible content."
        ),
        "global_rule": (
            "Define the requested target illumination, color balance, practical lighting, reflections, "
            "and atmosphere while preserving the scene layout and object identity."
        ),
        "local_rule": (
            "The target prompt must describe the exact local content under the requested time-of-day "
            "lighting. Change only the illumination and color of existing local content. A new light "
            "source, localized glow, reflection, or other secondary effect requires visible local "
            "evidence or an explicit user request. Do not introduce an environmental region merely "
            "because it exists elsewhere."
        ),
    },
    "Upscale / Detailer": {
        "operation_mode": "faithful_upscale",
        "direct_prompt": "Upscale this image with realistic, source-faithful detail.",
        "instruction": (
            "Analyze the complete image for identity, materials, and recurring detail, then direct "
            "each exact tile to recover realistic source-supported clarity without redesigning content."
        ),
        "global_rule": (
            "Identify the source medium, genuine detail patterns, compression or scaling artifacts, "
            "and continuity that should guide faithful restoration."
        ),
        "local_rule": (
            "The target prompt must improve clarity and material detail supported by the exact tile "
            "while preserving its existing lighting, colors, geometry, and content. Preserve the "
            "source's depth of field: out-of-focus areas stay out of focus. Never name an object you "
            "cannot clearly verify in these pixels; describe an uncertain small feature only by its "
            "color and softness."
        ),
    },
}


CAPTION_DETAIL_VALUES = {
    "Adaptive by Tile (recommended)": 65,
    "Simple": 30,
    "Complex": 80,
    "Detailed": 90,
    "Maximum (every visible item)": 100,
}
TILE_DETAIL_INSTRUCTIONS = {
    "Simple": (
        "Use one short, literal description of the main visible local content. Include only its "
        "identity, essential material or surface, and a boundary when needed. Do not add generic "
        "quality phrases or enlarge ambiguous details."
    ),
    "Adaptive by Tile (recommended)": (
        "Keep simple or uniform tiles short. For complex tiles, distinguish important visible regions "
        "by coarse position, material, visible color, surface, boundary, and distinctive pattern. "
        "When repeated "
        "elements matter, describe their visible individual scale, density, spacing, and approximate "
        "count in natural language; preserve unit scale, density, spacing, and layout. Never replace "
        "many small repeated elements with one enlarged feature."
    ),
    "Complex": (
        "Describe every important visible region and relationship using coarse position, material, "
        "visible color, surface, boundary crossings, and distinctive patterns. Preserve "
        "repeated-element unit "
        "scale, density, spacing, approximate count, and partial-edge placement. Keep incidental "
        "microtexture brief and include only exact-tile evidence."
    ),
    "Detailed": (
        "Describe every important locally visible region and object using coarse position, material, "
        "visible color, surface finish, boundary crossings, and distinctive patterns. Describe "
        "repeated elements "
        "naturally when visible, including their individual scale, density, spacing, and approximate "
        "count; preserve unit scale, density, spacing, and layout. Never replace many small repeated "
        "elements with one enlarged feature. Include only details supported by the exact tile."
    ),
    "Maximum (every visible item)": (
        "List EVERY visible item in this exact tile - every object, structure, surface, plant, "
        "vehicle, fixture, and region - so a sampler that only renders what is named recreates "
        "exactly what is here. Give each item its identity, material, visible color, and coarse "
        "position. Preserve counts, spacing, and layout of repeated elements. Nothing visible may "
        "go unmentioned, but include only what these exact pixels support."
    ),
}


UNIFIED_TASK_INSTRUCTIONS = {
    "Google Image Enhance": """TASK: Google Image Enhance

Turn this Google Earth image into a realistic photograph. Keep everything exactly where it is - the same
buildings, streets, layout, and camera angle - and just make it look like a real, sharp photo.

Anything that looks broken, melted, smeared, or like debris is a capture glitch, not real damage: fix it
into how it should really look. Don't add anything that isn't clearly there, and never assume what kind
of place this is - describe only what is visible. Any big continuous area the image really has should
look the same everywhere it appears.""",
    "Style Transfer": """TASK: Style Transfer

Restyle this image into the look you describe in User Request (for example a painting style, a film look,
or a material change). Keep the scene, shapes, and layout the same - only the style changes.

Apply the same look consistently across the whole image, and don't add objects or effects that aren't
actually there.""",
    "Time of Day": """TASK: Time of Day

Change this image to the time of day you describe in User Request (for example day to night, or golden
hour). Keep everything in place - same buildings, layout, and camera angle - and only change the light,
color, and mood.

Don't invent new lights, reflections, or objects unless they are actually visible in the image.""",
    "Upscale / Detailer": """TASK: Upscale / Detailer

Sharpen this image and add realistic detail without changing it. Keep the same content, colors, lighting,
and layout - just recover clean, believable detail.

Blur is part of the photo: out-of-focus areas stay just as soft, and only what is sharp gets sharper.
Thin real details that cross blurred areas are content, not blur, however fine: name them and keep
every one. If something is too blurry or small to identify for certain, leave
it as soft shapes and colors - never guess what it might be. Don't redesign anything or add objects
that aren't there.""",
}
def _load_task_preset_file():
    """Let presets/task_presets.json override the embedded task instructions.

    Built-in presets ship as an editable data file so instruction packs can be
    improved or distributed separately without touching code. The embedded text
    above is the safety fallback when the file is missing or unreadable.
    """
    path = Path(__file__).resolve().parents[1] / "presets" / "task_presets.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    presets = payload.get("presets") if isinstance(payload, dict) else None
    if not isinstance(presets, dict):
        return
    for name, entry in presets.items():
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("instructions", "")).strip()
        if str(name) in UNIFIED_TASK_INSTRUCTIONS and text:
            UNIFIED_TASK_INSTRUCTIONS[str(name)] = text


_load_task_preset_file()


CONTEXT_AWARENESS_CHOICES = (
    "Local Evidence Only",
    "Scene Type + Continuity (recommended)",
    "Recognized Names When Visible",
)
# The only two user controls are the preset (complete instructions) and the
# Prompt Detail Level, which lives OUTSIDE the instructions and drives how much
# verified local detail each tile prompt carries. Everything else — blur policy,
# color naming, position wording — belongs inside the preset instructions, where
# it speaks with one voice and cannot be contradicted by a toggle.
TILE_DETAIL_CHOICES = (
    "Simple",
    "Adaptive by Tile (recommended)",
    "Complex",
    "Detailed",
    "Maximum (every visible item)",
)
SAMPLER_STYLE_CHOICES = (
    "Instruction edit (Klein, Qwen Edit)",
    "Plain description (SDXL, Flux, denoise)",
)

# A rare-use repair, not a style choice. Off is a true bypass: it changes
# nothing at all. On deletes color names from the finished tile prompt, for the
# one situation it exists to fix - a model tinting whole tiles a flat shade.
TILE_COLOR_CHOICES = (
    "Off - change nothing (default)",
    "On - delete color names from tile prompts",
)
# Older graphs and presets stored these labels; they must keep working.
_TILE_COLOR_LEGACY_OFF = "automatic (follow the preset instructions)"
_TILE_COLOR_LEGACY_ON = "no color words (keep original colors)"


def _color_words_mode(value):
    """Map the Color Words control to `keep` or `strip`, defaulting to keep.

    Anything unrecognised - a blank, an older label, a future relabel - means
    keep. A prompt-altering guard must never switch itself on by accident.
    """
    text = " ".join(str(value or "").split()).casefold()
    if not text or text.startswith("off") or text == _TILE_COLOR_LEGACY_OFF:
        return "keep"
    if (
        text == _TILE_COLOR_LEGACY_ON
        or text.startswith("on ")
        or text.startswith("on-")
        or "delete color" in text
        or "no color word" in text
    ):
        return "strip"
    return "keep"


def _tile_detail_contract(
    detail_mode,
    evidence_class,
    visual_complexity,
    tile_prompt_instruction="",
):
    """Turn a human detail choice into a predictable per-tile caption contract."""
    normalized = str(detail_mode or "").strip()

    visible_rule = str(tile_prompt_instruction or "").strip() or TILE_DETAIL_INSTRUCTIONS.get(
        normalized, TILE_DETAIL_INSTRUCTIONS["Adaptive by Tile (recommended)"]
    )

    if normalized == "Simple":
        return 28, visible_rule
    if normalized == "Complex":
        return 56, visible_rule
    if normalized == "Detailed":
        return 68, visible_rule
    if normalized == "Maximum (every visible item)":
        return 100, visible_rule

    if evidence_class == "uniform":
        return 18, visible_rule
    if evidence_class == "sparse" or visual_complexity == "simple":
        return 28, visible_rule
    if visual_complexity == "moderate":
        return 44, visible_rule
    return 68, visible_rule


def _parse_false_detections(value):
    if isinstance(value, (list, tuple)):
        candidates = value
    else:
        candidates = re.split(r"[,;\n]+", str(value or ""))
    result = []
    for candidate in candidates:
        term = re.sub(r"^\s*(?:no|not)\s+", "", str(candidate), flags=re.IGNORECASE)
        term = term.strip().strip(".\"'")
        if term and term.casefold() not in {item.casefold() for item in result}:
            result.append(term)
    return result


def _unified_setting(instructions, label):
    match = re.search(
        rf"^\s*{re.escape(label)}\s*:\s*(.+?)\s*$",
        str(instructions or ""),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def _unified_task_name(instructions):
    setting = _unified_setting(instructions, "TASK").casefold()
    for name in TASK_PRESET_CHOICES:
        if setting.startswith(name.casefold()):
            return name
    text = str(instructions or "").casefold()
    if "google earth" in text or "3d map" in text or "photogrammetr" in text:
        return "Google Image Enhance"
    if "time of day" in text or "day to night" in text or "nighttime" in text:
        return "Time of Day"
    if "style transfer" in text or "target style" in text:
        return "Style Transfer"
    return "Upscale / Detailer"


def _unified_analysis_mode(instructions):
    setting = _unified_setting(instructions, "IMAGE ANALYSIS").casefold()
    if any(term in setting for term in ("user request only", "no image analysis", "bypass")):
        return "Use User Request only (no image analysis)"
    return "Analyze whole image + exact tiles"


def _unified_context_awareness(instructions):
    setting = _unified_setting(instructions, "GLOBAL CONTEXT").casefold()
    if "local" in setting and "only" in setting:
        return "Local Evidence Only"
    if "recogn" in setting or "name" in setting:
        return "Recognized Names When Visible"
    return "Scene Type + Continuity (recommended)"


_UNIFIED_CONTROL_LABELS = (
    "TASK",
    "PROMPT DETAIL",
    "IMAGE ANALYSIS",
    "GLOBAL CONTEXT",
)


def _complete_unified_instructions(instructions):
    """Recover the built-in body when a saved workflow contains only control headers."""
    supplied = str(instructions or "").strip()
    task_name = _unified_task_name(supplied)
    if not supplied:
        return UNIFIED_TASK_INSTRUCTIONS[task_name]

    body_lines = []
    for line in supplied.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(
            re.match(rf"^{re.escape(label)}\s*:", stripped, re.IGNORECASE)
            for label in _UNIFIED_CONTROL_LABELS
        ):
            continue
        body_lines.append(stripped)
    if body_lines:
        return supplied

    completed = UNIFIED_TASK_INSTRUCTIONS[task_name]
    # Preserve any deliberate header choices from the saved workflow while
    # restoring the missing explanatory body from the matching built-in preset.
    for label in _UNIFIED_CONTROL_LABELS:
        value = _unified_setting(supplied, label)
        if value:
            completed = re.sub(
                rf"^\s*{re.escape(label)}\s*:.*$",
                f"{label}: {value}",
                completed,
                count=1,
                flags=re.IGNORECASE | re.MULTILINE,
            )
    return completed


def _remove_false_detections(value, excluded_terms):
    text = str(value or "")
    for term in excluded_terms:
        for form in _false_detection_forms(term):
            pattern = re.compile(rf"(?<!\w){re.escape(form)}(?!\w)", re.IGNORECASE)
            text = pattern.sub("", text)
    # Removing a word out of a compound leaves debris ("wire-like" -> "-like").
    text = re.sub(r"(?<!\w)-+(?=\w)", "", text)
    text = re.sub(r"\s+([,;.])", r"\1", text)
    text = re.sub(r"([,;])(?:\s*[,;])+", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,;:-")


_CAMERA_VIEW_TOKEN = re.compile(
    r"\b(?:aerial|overhead|top-down|high-angle|low-angle|street-level|ground-level|"
    r"oblique|bird(?:'s|-)?eye|close-up|macro)(?:\s+(?:view|perspective|shot))?\b|"
    r"\blooking\s+(?:down|up|across|toward(?:\s+the)?\s+camera)\b",
    re.IGNORECASE,
)


def _camera_view_only(value):
    """Keep camera geometry while discarding scene nouns from a VLM `view` field."""
    text = " ".join(str(value or "").split())
    matches = [match.group(0).strip() for match in _CAMERA_VIEW_TOKEN.finditer(text)]
    result = []
    for item in matches:
        normalized = item.casefold()
        if normalized not in {existing.casefold() for existing in result}:
            result.append(item)
    return " ".join(result)


_SPATIAL_TERMS = {
    "left": ("left", "left-hand", "western", "west"),
    "right": ("right", "right-hand", "eastern", "east"),
    "top": ("top", "upper", "northern", "north"),
    "bottom": ("bottom", "lower", "foreground", "southern", "south"),
    "center": ("center", "central", "middle"),
}
def _has_spatial_term(text, direction):
    return any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE)
        for term in _SPATIAL_TERMS[direction]
    )


def _entry_contains_excluded(text, excluded_terms):
    """True when any user-confirmed false term appears in a canonical entry.

    A known-false concept must drop the whole surface/object entry so it can never
    become a tile candidate, not merely be word-scrubbed into a degraded phrase.
    """
    blob = str(text or "")
    for term in excluded_terms or ():
        for form in _false_detection_forms(term):
            if re.search(rf"(?<!\w){re.escape(form)}(?!\w)", blob, re.IGNORECASE):
                return True
    return False


def _tile_spatial_coverage(tile, metadata, use_core=False):
    """Return coarse regions touched by the tile crop.

    ``use_core=True`` measures only the tile's core rectangle (without the
    blending overlap). Surface stamping uses it: at high scales the few-pixel
    overlap pokes a middle-row tile across the bottom band boundary, which
    stamped canonical "open water" onto blue mountain-mist tiles.
    """
    width = max(1.0, float(metadata.get("image_width", 1)))
    height = max(1.0, float(metadata.get("image_height", 1)))
    if use_core:
        tile_x = float(tile.get("core_x", tile.get("x", 0)))
        tile_y = float(tile.get("core_y", tile.get("y", 0)))
        tile_w = float(tile.get("core_width", tile.get("width", 0)))
        tile_h = float(tile.get("core_height", tile.get("height", 0)))
    else:
        tile_x = float(tile.get("x", 0))
        tile_y = float(tile.get("y", 0))
        tile_w = float(tile.get("width", 0))
        tile_h = float(tile.get("height", 0))
    left = tile_x / width
    top = tile_y / height
    right = (tile_x + tile_w) / width
    bottom = (tile_y + tile_h) / height
    horizontal = {
        name
        for name, start, end in (
            ("left", 0.0, 0.4),
            ("center", 0.4, 0.6),
            ("right", 0.6, 1.0),
        )
        if right > start and left < end
    }
    vertical = {
        name
        for name, start, end in (
            ("top", 0.0, 0.4),
            ("center", 0.4, 0.6),
            ("bottom", 0.6, 1.0),
        )
        if bottom > start and top < end
    }
    return horizontal, vertical


def _coarse_location_axes(value):
    text = str(value or "")
    horizontal = (
        "left"
        if _has_spatial_term(text, "left")
        else "right"
        if _has_spatial_term(text, "right")
        else "center"
        if _has_spatial_term(text, "center")
        else None
    )
    vertical = (
        "top"
        if _has_spatial_term(text, "top")
        else "bottom"
        if _has_spatial_term(text, "bottom")
        else "center"
        if _has_spatial_term(text, "center")
        else None
    )
    return horizontal, vertical


_COARSE_BANDS = {
    "left": (0.0, 0.4),
    "center": (0.4, 0.6),
    "right": (0.6, 1.0),
    "top": (0.0, 0.4),
    "bottom": (0.6, 1.0),
}


def _normalized_tile_core_rect(tile, metadata):
    width = max(1.0, float(metadata.get("image_width", 1)))
    height = max(1.0, float(metadata.get("image_height", 1)))
    x = float(tile.get("core_x", tile.get("x", 0)))
    y = float(tile.get("core_y", tile.get("y", 0)))
    tile_width = float(tile.get("core_width", tile.get("width", 0)))
    tile_height = float(tile.get("core_height", tile.get("height", 0)))
    return (
        max(0.0, min(1.0, x / width)),
        max(0.0, min(1.0, y / height)),
        max(0.0, min(1.0, (x + tile_width) / width)),
        max(0.0, min(1.0, (y + tile_height) / height)),
    )


def _coarse_cell_overlap(tile_rect, horizontal, vertical):
    horizontal_band = _COARSE_BANDS.get(horizontal)
    vertical_band = _COARSE_BANDS.get(vertical)
    if horizontal_band is None or vertical_band is None:
        return 0.0
    left, top, right, bottom = tile_rect
    overlap_width = max(
        0.0, min(right, horizontal_band[1]) - max(left, horizontal_band[0])
    )
    overlap_height = max(
        0.0, min(bottom, vertical_band[1]) - max(top, vertical_band[0])
    )
    tile_area = max(1e-9, (right - left) * (bottom - top))
    return overlap_width * overlap_height / tile_area


def _contested_locations(entries):
    """Areas that two different continuous surfaces both claim.

    One patch of an image is open water or open sky, not both. When the brief
    claims both in the same area it has contradicted itself, and a stamped
    surface is authoritative - on an interior photo the model filed the window
    view as `sky` across all nine areas AND `water` across six, which handed a
    full-strength water claim to tiles showing hardwood floor.

    Neither claim wins: the contested areas are dropped from both, so those
    tiles fall back to their own pixels. Uncontested areas are untouched, so a
    normal landscape - sky on top, water below - is unaffected.
    """
    claims = {}
    for entry in entries[:12]:
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("id", "")).strip().casefold()
        locations = entry.get("locations")
        if not entry_id or not isinstance(locations, list):
            continue
        for location in locations[:9]:
            axes = _coarse_location_axes(str(location)[:40])
            if axes[0] and axes[1]:
                claims.setdefault(axes, set()).add(entry_id)
    return {axes for axes, owners in claims.items() if len(owners) > 1}


def _surface_map_context(
    global_context, tile, metadata, excluded_terms=(), limit=2, map_key="surface_map"
):
    """Return canonical continuous surfaces whose explicit locations touch this tile."""
    if not isinstance(global_context, dict):
        return []
    entries = global_context.get(map_key)
    if not isinstance(entries, list):
        return []

    # Core rectangle only: a surface identity is stamped as authoritative, so
    # the blending overlap must never pull a neighbor row's surface onto a
    # tile whose real content sits outside that surface's band.
    tile_horizontal, tile_vertical = _tile_spatial_coverage(tile, metadata, use_core=True)
    tile_rect = _normalized_tile_core_rect(tile, metadata)
    contested = _contested_locations(entries) if map_key == "surface_map" else set()
    ranked = []
    for order, entry in enumerate(entries[:12]):
        if not isinstance(entry, dict):
            continue
        raw_entry_text = " ".join(
            str(entry.get(field, ""))
            for field in ("id", "identity", "source_appearance", "target_prompt")
        )
        if _entry_contains_excluded(raw_entry_text, excluded_terms):
            continue
        surface_id = _remove_false_detections(str(entry.get("id", ""))[:40], excluded_terms)
        identity = _remove_false_detections(
            str(entry.get("identity", ""))[:160], excluded_terms
        )
        source_appearance = _remove_false_detections(
            str(entry.get("source_appearance", ""))[:220], excluded_terms
        )
        target_prompt = _remove_false_detections(
            str(entry.get("target_prompt", ""))[:260], excluded_terms
        )
        if map_key == "surface_map":
            # Cleaned HERE, before the tile ever sees the candidate. Stripping it
            # only at resolve time was not enough: the tile is shown the phrase
            # and copies it into its own description, so "reflecting city lights"
            # still reached every open-water tile and each invented its own
            # highlights - which is what made the tile grid visible on the water.
            target_prompt = _strip_reflected_objects(target_prompt)
        locations = entry.get("locations")
        if not surface_id or not identity or not target_prompt or not isinstance(locations, list):
            continue

        matching_locations = []
        matched_cells = set()
        for location in locations[:9]:
            cleaned_location = _remove_false_detections(
                str(location)[:40], excluded_terms
            )
            horizontal, vertical = _coarse_location_axes(cleaned_location)
            if not horizontal or not vertical:
                continue
            if (horizontal, vertical) in contested:
                continue
            if horizontal not in tile_horizontal or vertical not in tile_vertical:
                continue
            matching_locations.append(cleaned_location)
            matched_cells.add((horizontal, vertical))
        if not matching_locations:
            continue
        spatial_overlap = sum(
            _coarse_cell_overlap(tile_rect, horizontal, vertical)
            for horizontal, vertical in matched_cells
        )

        cleaned = {
            "id": surface_id,
            "identity": identity,
            "source_appearance": source_appearance,
            "target_prompt": target_prompt,
            "matching_locations": matching_locations,
            # Kept in the private tile reference for deterministic resolution.
            # The instruction builder removes it before presenting candidates to
            # the vision model.
            "spatial_overlap": round(spatial_overlap, 6),
        }
        ranked.append((spatial_overlap, -order, cleaned))

    ranked.sort(reverse=True)
    return [item[2] for item in ranked[: max(0, int(limit))]]


def _material_map_context(
    global_context, tile, metadata, excluded_terms=(), limit=2, allow_atmosphere=False
):
    """Canonical textured materials (fabric, brickwork, foliage) touching this tile.

    Materials are offered to textured tiles as CANDIDATES only - the tile's own
    caption stays primary and the resolver never overrides it with a material.
    The shared wording is what keeps a scarf's weave identical across tiles.
    Entries failing surface-prompt safety (process words, nearby objects) are
    simply dropped: a bad material candidate degrades to today's behavior, it
    never triggers retries and never reaches a tile.

    The bar here is deliberately LOWER than the authoritative surface path. A
    30-tile coastal villa produced a brief whose every entry repeated its own
    identity ("rough white stone wall" -> "rough white stone wall"). Judged by
    the surface rule that is thin, so every material was dropped and the stone
    path came back worded differently in each tile it crossed. As advisory
    context that phrase is worth sharing; it can never take a tile over.
    """
    candidates = _surface_map_context(
        global_context, tile, metadata, excluded_terms, limit, map_key="material_map"
    )
    return [
        candidate
        for candidate in candidates
        if not _surface_prompt_problem(candidate, allow_atmosphere)
        and not _map_prompt_carries_nothing(candidate)
    ]


def _cleaned_map_entry(entry, excluded_terms):
    """Clean one surface/object map entry; None when invalid or user-excluded."""
    if not isinstance(entry, dict):
        return None
    raw_entry_text = " ".join(
        str(entry.get(field, ""))
        for field in ("id", "identity", "source_appearance", "target_prompt")
    )
    if _entry_contains_excluded(raw_entry_text, excluded_terms):
        return None
    cleaned = {
        "id": _remove_false_detections(str(entry.get("id", ""))[:40], excluded_terms),
        "identity": _remove_false_detections(str(entry.get("identity", ""))[:160], excluded_terms),
        "source_appearance": _remove_false_detections(
            str(entry.get("source_appearance", ""))[:220], excluded_terms
        ),
        "target_prompt": _remove_false_detections(
            str(entry.get("target_prompt", ""))[:260], excluded_terms
        ),
    }
    locations = entry.get("locations")
    if (
        not cleaned["id"]
        or not cleaned["identity"]
        or not cleaned["target_prompt"]
        or not isinstance(locations, list)
    ):
        return None
    cleaned["locations"] = [
        _remove_false_detections(str(location)[:40], excluded_terms)
        for location in locations[:9]
    ]
    parts = entry.get("parts")
    if isinstance(parts, dict):
        cleaned_parts = {}
        for label, text in list(parts.items())[:9]:
            text = _remove_false_detections(str(text)[:80], excluded_terms)
            if text:
                cleaned_parts[str(label)[:20]] = text
        if cleaned_parts:
            cleaned["parts"] = cleaned_parts
    return cleaned


def _component_axes(component, tiles_by_index, metadata):
    """Coarse (horizontal, vertical) cells covered by a measured uniform-tile run."""
    pairs = set()
    for tile_index in list(component.get("members", [])) + list(
        component.get("attached", [])
    ):
        tile = tiles_by_index.get(int(tile_index))
        if tile is None:
            continue
        horizontal, vertical = _tile_spatial_coverage(tile, metadata)
        for h in horizontal:
            for v in vertical:
                pairs.add((h, "middle" if v == "center" else v))
    return pairs


def _component_canonical_surface(global_context, component_axes_pairs, excluded_terms):
    """Pick the whole-image surface that best overlaps a measured uniform-tile run.

    A run of adjacent same-color uniform tiles IS one continuous surface by pixel
    measurement. Every tile in the run therefore receives the same canonical
    surface entry — including tiles the vision model's location list missed — so
    a flat water tile can never be left without its "you are water" context.
    """
    if not isinstance(global_context, dict):
        return None
    entries = global_context.get("surface_map")
    if not isinstance(entries, list):
        return None
    best = None
    best_score = 0
    tied = False
    for entry in entries[:12]:
        cleaned = _cleaned_map_entry(entry, excluded_terms)
        if cleaned is None:
            continue
        entry_axes = {location_axes(location) for location in cleaned["locations"]}
        score = len(entry_axes & component_axes_pairs)
        if score > best_score:
            best_score = score
            tied = False
            matching = [
                location
                for location in cleaned["locations"]
                if location_axes(location) in component_axes_pairs
            ]
            best = {
                "id": cleaned["id"],
                "identity": cleaned["identity"],
                "source_appearance": cleaned["source_appearance"],
                "target_prompt": cleaned["target_prompt"],
                "matching_locations": matching or cleaned["locations"],
                "selection_source": "measured_uniform_region",
            }
        elif score and score == best_score and best is not None:
            if str(cleaned.get("id", "")).casefold() != str(best.get("id", "")).casefold():
                tied = True
    return None if tied else best


def _declared_surface_labels(global_context):
    """Names the whole-image pass already gave to a continuous surface.

    A region the brief calls "water" belongs to the water surface, not to an
    object that happens to list that region. Whole-image models routinely hand
    a picture-spanning "main subject" every region and then fill the water
    regions of its `parts` map with "water" - offering that back as an object
    part hint let a water tile confirm an entire skyline on the word "water".
    """
    labels = set()
    if not isinstance(global_context, dict):
        return labels
    for entry in global_context.get("surface_map") or []:
        if not isinstance(entry, dict):
            continue
        for key in ("id", "identity", "target_prompt"):
            value = str(entry.get(key, "")).strip().casefold()
            if value:
                labels.add(value)
    return labels


def _object_map_context(global_context, tile, metadata, excluded_terms=(), limit=2, field="object_map"):
    """Return canonical object candidates for this tile, with a per-tile part hint.

    Every object candidate is offered as a HYPOTHESIS to any tile that can hold
    discrete content — the tile must still confirm it from its own pixels before
    the resolver reuses the canonical phrase, so nothing here draws an object.
    Stated locations rank the candidates and pick the tile's `part_in_this_tile`
    hint, but they no longer gate the offer: a main subject whose location list
    missed one area (the camel's hump) must still be confirmable there.
    """
    if not isinstance(global_context, dict):
        return []
    entries = global_context.get(field)
    if not isinstance(entries, list):
        return []

    tile_horizontal, tile_vertical = _tile_spatial_coverage(tile, metadata)
    tile_pairs = {
        (h, "middle" if v == "center" else v)
        for h in tile_horizontal
        for v in tile_vertical
    }
    tile_rect = _normalized_tile_core_rect(tile, metadata)
    surface_labels = _declared_surface_labels(global_context)
    ranked = []
    for order, entry in enumerate(entries[:12]):
        cleaned = _cleaned_map_entry(entry, excluded_terms)
        if cleaned is None:
            continue
        parts = cleaned.get("parts") or {}
        # Regions the brief itself calls a declared surface are not this
        # object's, however many locations the object claimed.
        surface_claimed = {
            location_axes(label)
            for label, text in parts.items()
            if str(text).strip().casefold() in surface_labels
        }
        matching_locations = [
            location
            for location in cleaned["locations"]
            if location_axes(location) in tile_pairs
            and location_axes(location) not in surface_claimed
        ]
        overlap = len(matching_locations)
        matched_cells = {
            _coarse_location_axes(location) for location in matching_locations
        }
        spatial_overlap = sum(
            _coarse_cell_overlap(tile_rect, horizontal, vertical)
            for horizontal, vertical in matched_cells
            if horizontal and vertical
        )
        part_hints = []
        for label, text in parts.items():
            if location_axes(label) not in tile_pairs:
                continue
            if str(text).strip().casefold() in surface_labels:
                continue
            if text not in part_hints:
                part_hints.append(text)
        candidate = {
            "id": cleaned["id"],
            "identity": cleaned["identity"],
            "source_appearance": cleaned["source_appearance"],
            "target_prompt": cleaned["target_prompt"],
            "spatial_overlap": round(spatial_overlap, 6),
        }
        if part_hints:
            candidate["part_in_this_tile"] = "; ".join(part_hints[:3])
        ranked.append((overlap, -order, candidate))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked[: max(0, int(limit))]]


def _request_fragment(value):
    """Turn a full user command into a short phrase that can follow a preset edit action."""
    text = str(value or "").strip().rstrip(" .")
    prefixes = (
        "convert this image to ",
        "convert this image into ",
        "convert the image to ",
        "convert the image into ",
        "change this image to ",
        "change this image into ",
        "turn this image into ",
        "make this image ",
    )
    lowered = text.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip().rstrip(" .")
    return text


def _edit_action(preset_name, preset, user_request):
    """Build the concise imperative that the image-edit model sees first."""
    user = _request_fragment(user_request)
    if not user:
        return preset["direct_prompt"]
    if preset_name == "Style Transfer":
        return f"Convert this image to {user}."
    if preset_name == "Time of Day":
        return f"Change this image to {user}."
    return f"{preset['direct_prompt'].rstrip(' .')}. {user}."


def _prompt_blueprint(global_instruction, prompt_system):
    instructions = str(prompt_system.get("instructions", "")).strip()
    user_request = str(prompt_system.get("user_request", "")).strip()
    false_detections = prompt_system.get("known_false_detections", [])
    edit_action = str(prompt_system.get("edit_action", "")).strip()
    detail_name = str(prompt_system.get("caption_detail", ""))
    task_preset = str(prompt_system.get("task_preset", ""))
    task_step3_rule = (
        "Anything that looks broken or like debris is treated as a repair, not copied. "
        if task_preset == "Google Image Enhance"
        else "Blur, color, and every other style choice follow the preset instructions. "
    )
    suffix = str(prompt_system.get("prompt_suffix", "")).strip().strip(",.")
    suffix_line = f", {suffix}" if suffix else ""
    if str(prompt_system.get("prompt_format", "instruction_edit")) == "description":
        step4 = (
            f"<the verified description for this tile>{suffix_line}.\n\n"
            "Your preset is set for a DENOISE model, so the tile description is the whole prompt - "
            "no command is put in front of it. Denoise models paint every word they are given."
        )
    else:
        step4 = (
            f"{edit_action.rstrip(' .')}. <the verified description for this tile>{suffix_line}.\n\n"
            "Your preset is set for an EDIT model, so the tile description follows a direct command."
        )
    return f"""HOW YOUR PROMPTS ARE BUILT

Your editable inputs
- Instructions: {instructions or '[blank]'}
- User Request: {user_request or '[blank]'}
- Known False Detections (things you know are NOT there; removed at every stage): {', '.join(false_detections) or '[blank]'}
- Prompt Detail Level: {detail_name}

Step 1 - Look at the whole image once
The enlarged image is analyzed a single time to produce a short scene brief: what the scene is, where
it likely is, the camera angle, a map of what sits in each area, and one fixed description for each big
surface (water, sky, ground, roads) and each large object that crosses tiles. Big smooth areas are also
found by direct pixel measurement, so the brief can never skip one: if the analysis misses a measured
area, it is asked again until every real surface is named.

Step 2 - Choose what each tile needs
For every tile, only the surfaces, objects, and areas that actually overlap that tile are passed on.
Touching smooth tiles with the same color are treated as one continuous surface, so every tile in it
shares the same identity and description. A big surface only takes over a whole tile when it really
covers that tile; where it just clips a corner, the tile keeps its own description. A scene label may
only claim a big surface (like water) when the surface list actually contains it.

Step 3 - Describe each tile on its own
Each tile is described using only what its own pixels show, plus the matching context from Step 1. A big
surface reuses the exact same wording in every tile it touches, so water, sky, and ground stay identical.
{task_step3_rule}Known False Detections are removed again here.

Step 4 - Final prompt sent to the image model
{step4}

The image model is a separate, replaceable block."""


def _prompt_system_value(prompt_system, key, default=""):
    if isinstance(prompt_system, dict):
        return prompt_system.get(key, default)
    return default


def _clean_json_response(value):
    text = str(value or "").strip()
    while text.endswith("<end_of_turn>"):
        text = text[: -len("<end_of_turn>")].rstrip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    return text


def _global_context_data(global_context):
    parsed = _loads_tolerant(global_context)
    if parsed is not None:
        return parsed
    text = _clean_json_response(global_context)
    return {"raw_context": text or "[No full-image scene context available]"}


_SURFACE_CLAIM_WORDS = re.compile(
    r"\b(?:water(?:front|side)?|harbou?r(?:front)?|lake(?:front|side)?|"
    r"river(?:front|side)?|sea(?:side)?|ocean(?:front)?|coast(?:al|line)?|"
    r"bay|beach|shore(?:line)?|marina|pond)\b",
    re.IGNORECASE,
)


def _licensed_scene_type(scene_type, global_context):
    """A scene label may claim a big surface only when the surface map backs it up.

    The surface map is grounded (pixel-measured and validated), so it is the license
    for surface words in delivered context. A hallucinated "waterfront" scene label on
    an image whose surface map contains no water would otherwise flow into tile
    prompts and paint ponds into parks.
    """
    text = str(scene_type or "")
    if not _SURFACE_CLAIM_WORDS.search(text):
        return text
    surface_map = (
        global_context.get("surface_map") if isinstance(global_context, dict) else None
    )
    licensed = ""
    if isinstance(surface_map, list):
        licensed = " ".join(
            f"{entry.get('identity', '')} {entry.get('id', '')}"
            for entry in surface_map
            if isinstance(entry, dict)
        )
    if _SURFACE_CLAIM_WORDS.search(licensed):
        return text
    cleaned = _SURFACE_CLAIM_WORDS.sub("", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" ,;:-")


def _compact_global_context(
    global_context,
    evidence_class="structured",
    context_awareness="Scene Type + Continuity (recommended)",
    excluded_terms=(),
    task_preset="",
):
    if not isinstance(global_context, dict):
        return _remove_false_detections(str(global_context)[:1400], excluded_terms)
    compact = {}
    # Pass only tiny whole-image context that helps a tile read an ambiguous crop.
    # A UNIFORM (smooth) tile is defined entirely by its own pixels; giving it the scene
    # label or place name makes a flat water/sky tile read as "urban waterfront" or a city
    # name, which is a false detection. So a uniform tile gets camera geometry ONLY. A
    # structured tile gets full context to disambiguate a complex crop; a sparse tile gets
    # light scene context for its small edge feature.
    awareness = str(context_awareness or "Scene Type + Continuity (recommended)")
    if awareness != "Local Evidence Only":
        if evidence_class == "uniform":
            keys = ("view",)
        elif evidence_class == "sparse":
            keys = ("scene_type", "view")
        else:
            keys = ("scene_type", "geographic_context", "view")
        for key in keys:
            value = global_context.get(key)
            if not value:
                continue
            cleaned = _remove_false_detections(str(value)[:300], excluded_terms)
            if key == "view":
                cleaned = _camera_view_only(cleaned)
            elif key == "scene_type":
                cleaned = _licensed_scene_type(cleaned, global_context)
            if cleaned:
                compact[key] = cleaned
    return json.dumps(compact, ensure_ascii=True, separators=(",", ":"))


def _local_pixel_evidence(tile_image):
    image = tile_image[0] if tile_image.ndim == 4 else tile_image
    rgb = image[..., :3].movedim(-1, 0).unsqueeze(0)
    height, width = image.shape[:2]
    scale = min(1.0, 256.0 / max(height, width))
    if scale < 1.0:
        rgb = F.interpolate(
            rgb,
            size=(max(2, round(height * scale)), max(2, round(width * scale))),
            mode="area",
        )
    rgb = rgb[0]
    horizontal = (rgb[:, :, 1:] - rgb[:, :, :-1]).abs().amax(dim=0).flatten()
    vertical = (rgb[:, 1:, :] - rgb[:, :-1, :]).abs().amax(dim=0).flatten()
    edges = torch.cat((horizontal, vertical))
    standard_deviation = float(rgb.std(dim=(1, 2)).amax().item())
    edge_fraction = float((edges > 0.03).float().mean().item())
    if standard_deviation < 0.025 and edge_fraction < 0.005:
        evidence_class = "uniform"
        visual_complexity = "simple"
        classification = (
            "UNIFORM REGION: local pixels do not independently resolve semantic identity. "
            "Use compact global context plus the exact tile's color and texture to identify the "
            "continuous region without importing discrete objects from elsewhere."
        )
    elif edge_fraction < 0.04:
        evidence_class = "sparse"
        visual_complexity = "simple"
        classification = (
            "SPARSE BOUNDARY CONTENT: inspect the crop edges for a small partial object and state its "
            "exact local position; do not expand it into a scene-level object inventory."
        )
    else:
        evidence_class = "structured"
        if edge_fraction >= 0.15:
            visual_complexity = "dense"
        elif edge_fraction >= 0.08:
            visual_complexity = "complex"
        else:
            visual_complexity = "moderate"
        classification = "STRUCTURED LOCAL CONTENT: caption only object boundaries visible in this exact tile."
    metrics = (
        f"{classification} Deterministic complexity: {visual_complexity} "
        f"(edge fraction {edge_fraction:.3f})."
    )
    return evidence_class, visual_complexity, metrics


def _build_tile_instruction(
    tile,
    metadata,
    global_context,
    prompt_system,
    evidence_class,
    visual_complexity,
    pixel_evidence,
):
    prompt_strategy = str(
        _prompt_system_value(prompt_system, "prompt_strategy", "task_directed")
    )
    user_instruction = str(
        _prompt_system_value(prompt_system, "user_instruction", "Faithfully enhance the source.")
    ).strip()
    workflow_instruction = str(
        _prompt_system_value(prompt_system, "workflow_instruction", SIMPLE_MASTER_INSTRUCTION)
    ).strip()
    task_preset = str(_prompt_system_value(prompt_system, "task_preset", "Custom"))
    direct_prompt = str(_prompt_system_value(prompt_system, "direct_prompt", "")).strip()
    edit_action = str(
        _prompt_system_value(prompt_system, "edit_action", direct_prompt or user_instruction)
    ).strip()
    local_task_rule = str(_prompt_system_value(prompt_system, "local_task_rule", "")).strip()
    detail_mode = str(
        _prompt_system_value(
            prompt_system, "caption_detail", "Adaptive by Tile (recommended)"
        )
    )
    tile_prompt_instruction = str(
        _prompt_system_value(prompt_system, "tile_prompt_instruction", "")
    ).strip()
    evidence_word_limit, evidence_detail_rule = _tile_detail_contract(
        detail_mode,
        evidence_class,
        visual_complexity,
        tile_prompt_instruction,
    )
    context_awareness = str(
        _prompt_system_value(
            prompt_system,
            "context_awareness",
            "Scene Type + Continuity (recommended)",
        )
    )
    known_false_detections = _parse_false_detections(
        _prompt_system_value(prompt_system, "known_false_detections", [])
    )
    describe_only = (
        str(_prompt_system_value(prompt_system, "prompt_format", "instruction_edit"))
        == "description"
    )

    if prompt_strategy == "direct_user":
        return str(
            _prompt_system_value(prompt_system, "direct_prompt", user_instruction)
        ).strip()

    tile_index = int(tile["tile_index"])
    tile_id = f"T{tile_index + 1:03d}"
    core_x = tile.get("core_x", tile["x"])
    core_y = tile.get("core_y", tile["y"])
    core_width = tile.get("core_width", tile["width"])
    core_height = tile.get("core_height", tile["height"])
    image_width = max(1, int(metadata.get("image_width", 1)))
    image_height = max(1, int(metadata.get("image_height", 1)))
    normalized_bounds = (
        f"left={core_x / image_width:.3f}, top={core_y / image_height:.3f}, "
        f"right={(core_x + core_width) / image_width:.3f}, "
        f"bottom={(core_y + core_height) / image_height:.3f}"
    )
    context_text = _compact_global_context(
        global_context,
        evidence_class,
        context_awareness,
        known_false_detections,
        task_preset,
    )
    canonical_surfaces = tile.get("canonical_surfaces", [])
    canonical_objects = tile.get("canonical_objects", [])
    canonical_materials = tile.get("canonical_materials", [])
    if canonical_surfaces or canonical_objects or canonical_materials:
        try:
            context_payload = json.loads(context_text)
        except (TypeError, json.JSONDecodeError):
            context_payload = {}
        if isinstance(context_payload, dict):
            # Whole-image location labels stay out of the tile's context: the tile
            # model echoes them as in-tile positions ("open water at bottom center"
            # for water on the tile's left). Locations remain in the reference for
            # the audit log only.
            if canonical_surfaces:
                context_payload["canonical_surface_candidates"] = [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key
                        not in ("matching_locations", "spatial_overlap", "selection_source")
                    }
                    for candidate in canonical_surfaces
                ]
            if canonical_objects:
                context_payload["canonical_object_candidates"] = [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key
                        not in ("matching_locations", "spatial_overlap", "selection_source")
                    }
                    for candidate in canonical_objects
                ]
            if canonical_materials:
                context_payload["canonical_material_candidates"] = [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key
                        not in ("matching_locations", "spatial_overlap", "selection_source")
                    }
                    for candidate in canonical_materials
                ]
            context_text = json.dumps(
                context_payload, ensure_ascii=True, separators=(",", ":")
            )

    target_task = edit_action or user_instruction or direct_prompt or workflow_instruction
    sparse_rule = ""
    if evidence_class in ("uniform", "sparse"):
        sparse_rule = (
            "This tile is sparse or nearly uniform. Name the concrete visible surface or region, "
            "not a generic process phrase such as `continuous visible region`. "
            "When one `canonical_surface_candidates` entry matches these pixels, return its `id` "
            "as `surface_id` and copy its `target_prompt` exactly as `surface_prompt`. "
            "Prefer a supported identity over a type guessed from color. Carry the supported "
            "whole-image camera/view descriptor into both `local_caption` and `target_prompt` so "
            "the region retains its perspective. Never call a visible filled region merely a "
            "background. Never include the global scene inventory or any discrete object from the "
            "complete image. Keep an "
            "ambiguous partial edge neutral."
        )
        if evidence_class == "sparse":
            sparse_rule += (
                " Any feature cut by a tile edge must remain partial: state its exact edge "
                "position in `local_caption` and keep it clipped to that same edge, at the same "
                "visible size, in `target_prompt`. Never complete an edge fragment inside the tile."
            )
    structured_rule = ""
    if evidence_class == "structured":
        structured_rule = (
            "This tile has detailed structured content. In `target_prompt`, name each distinct "
            "visible structure or object, its literal visible material, and its coarse in-tile "
            "position. Preserve the measured count, spacing, scale, and layout of repeated "
            "elements. Never collapse mixed structured content into one generic surface label; "
            "describe only what these pixels establish."
        )
    context_rule = {
        "Local Evidence Only": (
            "Do not use scene identity or named-place context; the exact tile supplies all content vocabulary."
        ),
        "Scene Type + Continuity (recommended)": (
            "Use generic scene type, geographic context, and view to clarify ambiguous local evidence, but do not import an object or repeated-detail inventory."
        ),
        "Recognized Names When Visible": (
            "A verified name may be used only when this exact tile visibly matches that named place or object; otherwise omit the name."
        ),
    }.get(context_awareness, "Use context only when the exact tile supports it.")
    # An edit engine receives a command line before this text, so the tile only
    # has to supply verified content. A denoise engine receives NOTHING else:
    # `target_prompt` is the whole prompt, and telling that model a command will
    # be prepended makes it hold back the description the sampler needs.
    target_prompt_role = (
        (
            "`target_prompt` is the COMPLETE prompt for this tile - it is sent to the sampler on "
            "its own, with no command in front of it. Write it as a plain description of the "
            "finished tile, not as an instruction, and do not describe the transformation itself. "
            "Every word will be painted, so write only about things in the picture: never mention "
            "the tile, the crop, the frame, or their edges, and never state how something should "
            "be treated."
        )
        if describe_only
        else (
            "`target_prompt` supplies verified local content and detail for the direct edit command "
            "that will be added before it. Do not restate, decorate, or explain the transformation "
            "in `target_prompt`; the direct command already owns it."
        )
    )
    false_detection_rule = (
        "User-confirmed false detections: "
        + ", ".join(known_false_detections)
        + ". Do not report or use them in any JSON field."
        if known_false_detections
        else "No user-confirmed false detections were supplied."
    )
    return f"""WORKFLOW INSTRUCTION:
{workflow_instruction}

TASK PRESET: {task_preset}
TARGET TASK: {target_task}

Inspect exact tile {tile_id}. The supplied image is the exact unmodified output-scale tile, not a
locator, mask, placeholder, or empty background. It alone determines which objects, surfaces, and
boundaries are locally present. The task brief supplies source interpretation, geographic identity,
target intent, and continuity only; it never proves that an object elsewhere is inside this tile.

WHOLE-IMAGE REFERENCE FILTERED FOR THIS TILE: {context_text}
Local evidence check: {pixel_evidence}
Where this crop sits in the full image (background only - never write these words or numbers into any
answer field): normalized core {normalized_bounds}.
Context use: {context_rule}
{false_detection_rule}

First inspect the exact pixels, then apply the target task to that local content. {local_task_rule}
The whole-image reference is background knowledge, never content: do not repeat its words in
`local_caption` or `target_prompt` unless this exact tile's own pixels visibly show those things.
Any position words you write must state where things sit INSIDE this exact tile as you see them in
these pixels; never reuse a location from the whole-image reference.
Do not create a separate region or object from a color, reflection, or content outside the crop.
Apply only changes necessary for the target task. Never invent an unrequested supporting cause,
object, effect, environmental region, surface property, color, material, or condition.
A requested property change does not imply a new cause or secondary visual effect. Never mention the
sky, weather, sun, or atmosphere in `target_prompt` unless those pixels are visible inside this exact
tile and named in `local_caption`. Before returning,
compare every noun and visual effect in `target_prompt` against `local_caption` and TARGET TASK. Delete
anything that is neither visibly present in `local_caption` nor explicitly requested by TARGET TASK.
Copy the same supported unit scale, density, spacing, partial-edge placement, and boundary geometry from
`local_caption` into `target_prompt`; never enlarge or complete a partial feature.
{sparse_rule}
{structured_rule}

`canonical_surface_candidates` contains large continuous surfaces selected only by coarse location.
Confirm a candidate from these pixels before using it. If confirmed, return its exact `id` and copy its
`target_prompt` verbatim; do not substitute a different color, lighting, texture, or synonym. Put only
other locally visible objects or boundaries in `local_features`. A surface candidate never proves that
any discrete object is present.

`canonical_object_candidates` lists the scene's main subject and large objects as hypotheses.
Use one only when this exact tile's own pixels clearly show a part of that same object. A candidate may
include `part_in_this_tile`, naming which part of it is expected here — check the pixels against it and,
when it matches, describe THAT part specifically. If confirmed,
return its exact `id` as `object_id` and copy its `target_prompt` verbatim as `object_prompt` so the object
looks identical in every tile it crosses; keep any purely local visible detail in `local_features`. Name a
confirmed object plainly and with certainty in `local_caption` and `target_prompt` — never hedge with
wording like "-like" or "appears to be", and say which part of it this tile shows. `object_prompt` is the
shared record only: `target_prompt` still describes ONLY what is visible inside this exact tile — the
visible PART of the subject, never the whole subject. If the
object's pixels are not visible in this tile, leave `object_id` and `object_prompt` empty. An object
candidate is never proof that the object is present here.

`canonical_material_candidates` lists large continuous textured materials with the ONE shared wording
that keeps their look identical across tiles. When these pixels
visibly show that material, describe it inside `local_caption` and `target_prompt` using the candidate's
`target_prompt` wording — the same material name, weave or pattern, pattern scale, and color words — and
still describe everything else this tile shows yourself. A material candidate is context, never an
override and never proof of presence: ignore any candidate these pixels do not show.

`local_caption` records literal visible source content. `corrections_applied` records only changes
required by the target task. {target_prompt_role} For a repair task, describe the supported corrected
local structure and material. Otherwise carry forward stable local content and only target attributes
explicitly named by the user. Do not write a new camera view or standalone scene. It must contain only
locally supported objects and materials, use positive language, and stay under {evidence_word_limit} words.
{evidence_detail_rule}
For a tile with several distinct objects or regions, use coarse local positions and relationships;
do not pad a simple region.

Return compact JSON only with exactly these keys:
{{"local_caption":"literal visible local content", "dominant_region":"main visible continuous region", "visible_boundaries":"only boundaries visible here", "surface_id":"confirmed canonical surface id or empty", "surface_prompt":"exact canonical surface target_prompt or empty", "object_id":"confirmed canonical object id or empty", "object_prompt":"exact canonical object target_prompt or empty", "local_features":"other locally visible target features, excluding the canonical surface and object", "corrections_applied":"task changes required here", "target_prompt":"desired finished tile"}}"""


def _build_all_tile_jobs(upscaled_tiles, tile_metadata_json, global_context, prompt_system):
    metadata = _load_tile_metadata(tile_metadata_json)
    global_data = _global_context_data(global_context)
    tile_records = sorted(metadata["tiles"], key=lambda item: item["tile_index"])
    source_indexes = {int(item.get("source_index", 0)) for item in tile_records}
    if int(metadata.get("image_batch", len(source_indexes))) != 1 or len(source_indexes) != 1:
        raise ValueError(
            "Smart Upscaler builds scene-aware prompts for one source image at a time. "
            "Split image batches before the upscaler so objects and surfaces cannot "
            "cross between unrelated images."
        )
    if len(tile_records) != upscaled_tiles.shape[0]:
        raise ValueError(
            f"Planner metadata describes {len(tile_records)} tiles, "
            f"but the upscaled image batch contains {upscaled_tiles.shape[0]}."
        )

    excluded_terms = _parse_false_detections(
        _prompt_system_value(prompt_system, "known_false_detections", [])
    )
    context_awareness_value = str(
        _prompt_system_value(prompt_system, "context_awareness", "")
    )

    # First pass: deterministic pixel measurement for every tile.
    analyses = []
    output_tile_width = int(metadata.get("output_tile_width", upscaled_tiles.shape[2]))
    output_tile_height = int(metadata.get("output_tile_height", upscaled_tiles.shape[1]))
    scale_factor = float(metadata.get("scale_factor", 1.0))
    for tile in tile_records:
        tile_index = int(tile["tile_index"])
        tile_image = upscaled_tiles[tile_index : tile_index + 1]
        x0 = round(float(tile.get("x", 0)) * scale_factor)
        y0 = round(float(tile.get("y", 0)) * scale_factor)
        valid_width = min(
            output_tile_width,
            int(tile_image.shape[2]),
            max(
                1,
                round(
                    (float(tile.get("x", 0)) + float(tile.get("width", 0)))
                    * scale_factor
                )
                - x0,
            ),
        )
        valid_height = min(
            output_tile_height,
            int(tile_image.shape[1]),
            max(
                1,
                round(
                    (float(tile.get("y", 0)) + float(tile.get("height", 0)))
                    * scale_factor
                )
                - y0,
            ),
        )
        caption_image = tile_image[:, :valid_height, :valid_width, :]
        evidence_class, visual_complexity, pixel_evidence = _local_pixel_evidence(
            caption_image
        )
        analyses.append(
            {
                "tile": tile,
                "tile_image": tile_image,
                "caption_image": caption_image,
                "valid_output_size": {
                    "width": valid_width,
                    "height": valid_height,
                },
                "evidence_class": evidence_class,
                "visual_complexity": visual_complexity,
                "pixel_evidence": pixel_evidence,
                "mean_rgb": tile_mean_rgb(caption_image),
            }
        )

    # A run of adjacent same-color uniform tiles is one continuous surface by
    # measurement, so every tile of the run gets the same whole-image surface
    # identity — even tiles the vision model's own location list missed.
    component_surface_by_tile = {}
    if context_awareness_value != "Local Evidence Only":
        tiles_by_index = {int(a["tile"]["tile_index"]): a["tile"] for a in analyses}
        components = uniform_tile_components(
            [
                {
                    "tile_index": int(a["tile"]["tile_index"]),
                    "source_index": int(a["tile"].get("source_index", 0)),
                    "row": int(a["tile"]["row"]),
                    "column": int(a["tile"]["column"]),
                    "evidence_class": a["evidence_class"],
                    "mean_rgb": a["mean_rgb"],
                }
                for a in analyses
            ]
        )
        for component in components:
            pairs = _component_axes(component, tiles_by_index, metadata)
            surface = _component_canonical_surface(global_data, pairs, excluded_terms)
            if surface:
                for member_index in list(component["members"]) + list(
                    component["attached"]
                ):
                    component_surface_by_tile[int(member_index)] = surface

    tile_images = []
    tile_instructions = []
    tile_references = []
    for analysis in analyses:
        tile = analysis["tile"]
        tile_image = analysis["tile_image"]
        evidence_class = analysis["evidence_class"]
        visual_complexity = analysis["visual_complexity"]
        pixel_evidence = analysis["pixel_evidence"]
        tile_index = int(tile["tile_index"])
        tile_id = f"T{tile_index + 1:03d}"
        reference = {
            "tile_index": tile_index,
            "tile_id": tile_id,
            "display_number": tile_index + 1,
            "source_index": int(tile["source_index"]),
            "row": int(tile["row"]),
            "column": int(tile["column"]),
            "position": tile["position"],
            "evidence_class": evidence_class,
            "visual_complexity": visual_complexity,
            "source_rect": {
                "x": tile["x"],
                "y": tile["y"],
                "width": tile["width"],
                "height": tile["height"],
            },
            "output_tile_size": {
                "width": metadata.get("output_tile_width"),
                "height": metadata.get("output_tile_height"),
            },
            "valid_output_size": analysis["valid_output_size"],
        }
        canonical_surfaces = []
        canonical_objects = []
        canonical_materials = []
        if context_awareness_value != "Local Evidence Only":
            canonical_surfaces = _surface_map_context(
                global_data, tile, metadata, excluded_terms
            )
            component_surface = component_surface_by_tile.get(tile_index)
            if component_surface:
                canonical_surfaces = [component_surface] + [
                    candidate
                    for candidate in canonical_surfaces
                    if str(candidate.get("id", "")).casefold()
                    != str(component_surface.get("id", "")).casefold()
                ]
                reference["canonical_surface_source"] = "measured_uniform_region"
            if canonical_surfaces:
                reference["canonical_surfaces"] = canonical_surfaces
            # The validated brief places each surface family (sky, water) by
            # location. When the brief DOES contain a family but its locations
            # do not reach this tile, an ambiguous tile claiming that family is
            # contradicting the brief (blue mountain mist guessed as "water") -
            # record it so the resolver neutralizes the guess. A family the
            # brief never mentions stays unrestricted: brief completeness is
            # the measured-region machinery's job, not this guard's.
            if evidence_class in ("uniform", "sparse") and isinstance(
                global_data.get("surface_map"), list
            ):
                brief_text = " ".join(
                    f"{entry.get('identity', '')} {entry.get('target_prompt', '')}"
                    for entry in global_data.get("surface_map", [])
                    if isinstance(entry, dict)
                )
                delivered_text = " ".join(
                    f"{candidate.get('identity', '')} {candidate.get('target_prompt', '')}"
                    for candidate in canonical_surfaces
                )
                unlicensed_families = sorted(
                    family
                    for family, pattern in _SURFACE_FAMILY_PATTERNS.items()
                    if pattern.search(brief_text)
                    and not pattern.search(delivered_text)
                )
                if unlicensed_families:
                    reference["unlicensed_surface_families"] = unlicensed_families
            # Discrete spanning objects are offered to tiles that can actually contain
            # them. A uniform (smooth) tile has no discrete object, so it never receives
            # object candidates; this preserves the no-object-leak guarantee.
            if evidence_class != "uniform":
                canonical_objects = _object_map_context(
                    global_data, tile, metadata, excluded_terms
                )
                if canonical_objects:
                    reference["canonical_objects"] = canonical_objects
                # Textured continuous materials (fabric, brickwork, foliage) are
                # likewise candidates only: shared wording for cross-tile weave
                # consistency, pixel-gated by the tile, never an override.
                canonical_materials = _material_map_context(
                    global_data,
                    tile,
                    metadata,
                    excluded_terms,
                    allow_atmosphere=(
                        str(prompt_system.get("operation_mode", ""))
                        == "style_transform"
                    ),
                )
                if canonical_materials:
                    reference["canonical_materials"] = canonical_materials
        if (
            evidence_class in ("uniform", "sparse")
            and context_awareness_value != "Local Evidence Only"
            and global_data.get("view")
        ):
            reference["view_context"] = _camera_view_only(
                _remove_false_detections(str(global_data["view"])[:220], excluded_terms)
            )
            if not reference["view_context"]:
                reference.pop("view_context")
        tile_context = dict(tile)
        if canonical_surfaces:
            tile_context["canonical_surfaces"] = canonical_surfaces
        if canonical_objects:
            tile_context["canonical_objects"] = canonical_objects
        if canonical_materials:
            tile_context["canonical_materials"] = canonical_materials
        tile_images.append(tile_image)
        tile_instructions.append(
            _build_tile_instruction(
                tile_context,
                metadata,
                global_data,
                prompt_system,
                evidence_class,
                visual_complexity,
                pixel_evidence,
            )
        )
        tile_references.append(json.dumps(reference))
    return tile_images, tile_instructions, tile_references


class SmartPromptGuidance:
    """Internal prompt engine behind the Prompt Director node.

    Turns a task preset (or the unified editable instruction block) into the
    whole-image analysis instruction and the prompt_system contract that the
    Tile Job Director, caption nodes, and resolver consume. Not a menu node.
    """

    def build(
        self,
        task_preset="Google Image Enhance",
        workflow_instruction="",
        caption_detail="Adaptive by Tile (recommended)",
        user_request="",
        image_analysis="Analyze whole image + exact tiles",
        context_awareness="Scene Type + Continuity (recommended)",
        known_false_detections="",
        tile_prompt_instruction="",
        sampler_prompt_style="Instruction edit (Klein, Qwen Edit)",
        prompt_suffix="",
        unified_instruction_ui=False,
        tile_colors=TILE_COLOR_CHOICES[0],
    ):
        preset_name = str(task_preset)
        preset = TASK_PRESETS.get(preset_name, TASK_PRESETS["Google Image Enhance"])
        supplied_instruction = str(workflow_instruction).strip()
        built_in_instructions = {item["instruction"] for item in TASK_PRESETS.values()}
        if bool(unified_instruction_ui):
            task_instruction = supplied_instruction or UNIFIED_TASK_INSTRUCTIONS[preset_name]
        elif not supplied_instruction or (
            supplied_instruction in built_in_instructions
            and supplied_instruction != preset["instruction"]
        ):
            task_instruction = preset["instruction"]
        else:
            task_instruction = supplied_instruction
        user = str(user_request).strip()
        operation_mode = preset["operation_mode"]
        detail_name = (
            str(caption_detail)
            if str(caption_detail) in CAPTION_DETAIL_VALUES
            else "Adaptive by Tile (recommended)"
        )
        supplied_tile_instruction = str(tile_prompt_instruction).strip()
        built_in_tile_instructions = set(TILE_DETAIL_INSTRUCTIONS.values())
        if not supplied_tile_instruction or (
            supplied_tile_instruction in built_in_tile_instructions
            and supplied_tile_instruction != TILE_DETAIL_INSTRUCTIONS[detail_name]
        ):
            resolved_tile_instruction = TILE_DETAIL_INSTRUCTIONS[detail_name]
        else:
            resolved_tile_instruction = supplied_tile_instruction
        awareness = (
            str(context_awareness)
            if str(context_awareness) in CONTEXT_AWARENESS_CHOICES
            else "Scene Type + Continuity (recommended)"
        )
        excluded_terms = _parse_false_detections(known_false_detections)
        false_detection_instruction = (
            "The user confirms these concepts are not present: "
            + ", ".join(excluded_terms)
            + ". Do not report them, use them as context, or include them in any output field."
            if excluded_terms
            else "The user supplied no known false detections."
        )
        bypass_vlm = str(image_analysis) == "Use User Request only (no image analysis)"
        edit_action = _edit_action(preset_name, preset, user)
        target_task = edit_action
        prompt_detail_section = (
            ""
            if bool(unified_instruction_ui)
            else f"""PROMPT DETAIL INSTRUCTION:
{resolved_tile_instruction}
"""
        )
        prompt_detail_bridge = (
            "Apply the PROMPT DETAIL rule written in WORKFLOW INSTRUCTION when deciding how much "
            "scene-wide recurring detail is useful to carry into exact tiles."
            if bool(unified_instruction_ui)
            else "Use the visible prompt-detail instruction below when deciding how much scene-wide recurring detail is useful to carry into exact tiles."
        )
        preset_global_rule = "" if bool(unified_instruction_ui) else preset["global_rule"]
        artifact_repair_rule = (
            "Treat torn, melted, smeared, warped, stretched, broken-looking, or debris-like "
            "geometry and texture as source reconstruction artifacts to repair, never as real "
            "damage or real objects. "
            if preset_name == "Google Image Enhance"
            else ""
        )
        global_instruction = f"""Look at the whole image once and return a SHORT JSON brief the tiles use for context and consistency. Keep it tiny.

WORKFLOW:
{task_instruction}

TASK: {preset_name}
TARGET: {target_task}

Return only these keys:
- scene_type: one short factual phrase naming what is actually visible, built only from things whose pixels you can point to in this image. Name water, a waterfront, a lake, or a coastline ONLY when open water pixels are clearly visible; never infer them from haze, color, city type, or typical geography.
- geographic_context: the real city, place, or named region ONLY when landmarks, signs, shoreline, streets, or building relationships clearly agree (use "likely ..." when several cues agree); otherwise "". Never invent or guess a place; if more than one city could fit what you see, return "".
- view: the true camera position and angle only, stated in a few words drawn from terms like aerial, overhead, oblique, high-angle, street-level, eye-level, close-up, macro. Choose only what matches this image; otherwise "".
- surface_map: one entry for EVERY large continuous uniform region in the image — open water, open sky, or large uniform open ground (bare field, parking lot, sand, snow). If the image contains open water or open sky anywhere, it MUST have an entry here; missing one breaks the result. NEVER buildings, facades, rooftops, streets, sidewalks, tree canopy, or mixed developed ground; those are structured content each tile describes itself. Each item: `id`, `locations`, `identity`, `target_prompt`. `locations` lists EVERY area the region covers, using only these nine labels: top left, top center, top right, middle left, center, middle right, bottom left, bottom center, bottom right. `target_prompt` is ONE specific description of that surface after TARGET — its own color, texture, and light — that every tile of it reuses word for word. It must say MORE than `identity` already says. Naming the thing again, even with two or three words, carries no information and is discarded: add how the surface actually looks close up - its finish, how light sits on it, the scale of its grain or pattern. It must name the real material (water, sky, sand, asphalt), never call it a "region" or "area", and must describe ONLY the surface itself: no nearby objects, and nothing it reflects. Reflected things are objects too — naming any of them, however distant or however small, makes tiles that hold only this surface paint them, and every such tile invents its own, so the tiles stop matching. Describe the surface's own colour, texture and how light behaves across it, and stop there. When a MEASURED FLAT REGIONS list is provided below, it comes from direct pixel measurement and every listed region MUST be identified here. Leave `surface_map` empty ONLY when no large uniform region exists anywhere.
- material_map: one entry for EVERY large continuous textured material that spans more than one area — clothing fabric or knitwear, a brick or stone wall, grass, foliage, wood planking. These are textured surfaces whose weave and pattern must look identical in every tile that shows them. Never a discrete object (that belongs in object_map) and never a uniform region already in surface_map. Each item: `id`, `locations`, `identity`, `target_prompt`, with `locations` using the same nine labels. `target_prompt` is ONE definitive description of the material itself after TARGET — its material name, weave or pattern type, pattern scale (fine, medium, or coarse), and color — that every tile showing it reuses word for word. It must say MORE than `identity` already says. Naming the thing again, even with two or three words, carries no information and is discarded: add how the surface actually looks close up - its finish, how light sits on it, the scale of its grain or pattern. Describe only the material; never nearby objects. Leave `material_map` empty when no textured material spans areas.
- object_map: one entry for the image's MAIN SUBJECT, plus one for every other large discrete thing that crosses more than one area — an animal, a person, a vehicle, a building, a structure. The main subject ALWAYS gets an entry, including when the whole of it sits inside a single area; missing it makes every tile guess. Leave `object_map` empty only when the image genuinely has no discrete subject at all. `locations` lists EVERY area any part of it covers. `parts` MUST be a JSON object, never an array: each key is one listed location and each value is a short literal phrase naming the visible piece occupying that location. Never emit template tokens or placeholder values. `identity` states plainly and with certainty what it is — never hedge with wording like "-like" or "appears to be". `target_prompt` is ONE definitive description of that subject after TARGET that every tile touching it builds on. It must say MORE than `identity` already says - the subject's materials and how they look close up - because naming it again carries no information. Never a small or repeated item, and never a surface.

{artifact_repair_rule}Describe only what the image supports; never invent content. {false_detection_instruction}
{preset_global_rule}
Return compact JSON with exactly these keys: scene_type, geographic_context, view, surface_map, material_map, object_map."""
        prompt_system = {
            "task_preset": preset_name,
            "workflow_instruction": task_instruction,
            "instructions": task_instruction,
            "user_request": user,
            "user_instruction": user,
            "operation_mode": operation_mode,
            "direct_prompt": target_task,
            "edit_action": edit_action,
            # The preset's per-tile expertise always ships to the tile stage; the
            # unified UI hides it from the editable text but must not lose it.
            "local_task_rule": preset["local_rule"],
            "prompt_strategy": "direct_user" if bypass_vlm else "task_directed",
            "composition_mode": "direct_user" if bypass_vlm else "task_directed",
            "caption_detail": detail_name,
            "tile_prompt_instruction": resolved_tile_instruction,
            "context_awareness": awareness,
            "known_false_detections": excluded_terms,
            "prompt_format": (
                "description"
                if "description" in str(sampler_prompt_style).casefold()
                else "instruction_edit"
            ),
            "prompt_suffix": str(prompt_suffix).strip(),
            "color_words": _color_words_mode(tile_colors),
            "unified_instruction_ui": bool(unified_instruction_ui),
        }
        if bool(unified_instruction_ui):
            combined = _prompt_blueprint(global_instruction, prompt_system)
            prompt_system["prompt_blueprint"] = combined
        else:
            combined = (
                f"TASK PRESET: {preset_name}\n"
                f"WORKFLOW INSTRUCTION: {task_instruction}\n"
                f"EDIT ACTION: {edit_action}\n"
                f"CAPTION DETAIL: {detail_name}\n"
                f"TILE PROMPT INSTRUCTION: {resolved_tile_instruction}\n"
                f"GLOBAL CONTEXT: {awareness}\n"
                f"KNOWN FALSE DETECTIONS (VLM ONLY): {', '.join(excluded_terms) or '[blank]'}\n"
                f"USER REQUEST: {user or '[blank]'}\n"
                f"IMAGE ANALYSIS: {image_analysis}"
            )
        return global_instruction, prompt_system, combined


class SmartUnifiedPromptGuidance:
    """Three-field Prompt Director used by the current general workflow."""

    CATEGORY = "Smart Upscaler/Prompting"
    RETURN_TYPES = ("STRING", "SMART_PROMPT_SYSTEM", "STRING")
    RETURN_NAMES = ("global_instruction", "prompt_system", "combined_instructions")
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instructions": (
                    "STRING",
                    {
                        "default": UNIFIED_TASK_INSTRUCTIONS["Google Image Enhance"],
                        "multiline": True,
                        "label": "1. Instructions - what this job is",
                        "tooltip": "The complete job description, normally filled in by Load Preset at the bottom of this node. TASK, IMAGE ANALYSIS, and GLOBAL CONTEXT settings are written inside this one box.",
                    },
                ),
                "user_request": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "label": "2. Your request (optional)",
                        "tooltip": "What you want changed - a style, a time of day, a specific edit. Leave blank when the Instructions already describe the whole job.",
                    },
                ),
                "known_false_detections": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "label": "3. Not in this image (optional)",
                        "tooltip": "Comma-separated things you know are NOT in the picture. They are kept out of the analysis and out of every tile prompt; they never become negative prompts.",
                    },
                ),
            },
            # Optional so a workflow saved before this control existed still
            # queues cleanly (a missing optional value simply uses the default).
            "optional": {
                "prompt_suffix": (
                    "STRING",
                    {
                        "default": "Fine detail",
                        "multiline": True,
                        "label": "4. Tile prompts: extra words at the end",
                        "tooltip": "Plain words added to the end of every tile prompt - not an instruction, just text the sampler reads (quality tags, a style word). Leave empty for none.",
                    },
                ),
                "tile_detail": (
                    list(TILE_DETAIL_CHOICES),
                    {
                        "default": "Adaptive by Tile (recommended)",
                        "label": "5. Tile prompts: how much detail",
                        "tooltip": "How much verified local detail each tile prompt carries. Simple: one short literal line. Adaptive: short for plain tiles, rich for complex ones. Complex/Detailed: full materials, colors, and patterns. Maximum: name every visible item - made for samplers that only render what is named. Lower settings are faster.",
                    },
                ),
                # Hidden in the UI: the Load Preset dropdown sets this from the
                # preset's model family (edit vs denoise). Kept as a real input
                # so the value serializes and the engine contract is unchanged.
                "sampler_prompt_style": (
                    list(SAMPLER_STYLE_CHOICES),
                    {
                        "default": "Instruction edit (Klein, Qwen Edit)",
                        "label": "Sampler Prompt Style",
                        "tooltip": "Set automatically by the chosen preset's model family. Edit models (Klein, Qwen Edit) get an edit command plus the tile description. Denoise models (SDXL, Flux) get a plain complete description and use the negative prompt.",
                    },
                ),
                "tile_colors": (
                    list(TILE_COLOR_CHOICES),
                    {
                        "default": TILE_COLOR_CHOICES[0],
                        "label": "6. Repair: delete color names (rare)",
                        "tooltip": "Leave this Off - Off changes nothing at all. Turn it On only for the one problem it fixes: a model painting whole tiles a flat shade (the checkerboard tint), which happens when tile prompts name colors. On deletes color names from the finished tile prompt. Colors you typed into Your Request are always kept.",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(
        cls, tile_detail=None, sampler_prompt_style=None, tile_colors=None
    ):
        # Accept any stored values so graphs saved with older control layouts
        # still queue; build() maps unknown values to safe defaults.
        return True

    def build(
        self,
        instructions="",
        user_request="",
        known_false_detections="",
        tile_detail="Adaptive by Tile (recommended)",
        sampler_prompt_style="Instruction edit (Klein, Qwen Edit)",
        prompt_suffix="Fine detail",
        tile_colors=TILE_COLOR_CHOICES[0],
    ):
        # Graphs saved before the suffix box moved above the dropdowns carry
        # their optional widget values one slot late (values apply by position).
        # A dropdown choice arriving in the suffix box identifies that layout
        # deterministically; shift everything back and restore the default.
        if str(prompt_suffix) in TILE_DETAIL_CHOICES:
            shifted_style = str(tile_detail)
            tile_detail = str(prompt_suffix)
            prompt_suffix = "Fine detail"
            if shifted_style in SAMPLER_STYLE_CHOICES:
                sampler_prompt_style = shifted_style
        elif str(prompt_suffix) in SAMPLER_STYLE_CHOICES:
            sampler_prompt_style = str(prompt_suffix)
            prompt_suffix = "Fine detail"
        complete_instructions = _complete_unified_instructions(instructions)
        task_name = _unified_task_name(complete_instructions)
        detail_name = (
            str(tile_detail)
            if str(tile_detail) in TILE_DETAIL_CHOICES
            else "Adaptive by Tile (recommended)"
        )
        return SmartPromptGuidance().build(
            task_preset=task_name,
            workflow_instruction=complete_instructions,
            caption_detail=detail_name,
            user_request=user_request,
            image_analysis=_unified_analysis_mode(complete_instructions),
            context_awareness=_unified_context_awareness(complete_instructions),
            known_false_detections=known_false_detections,
            tile_prompt_instruction=TILE_DETAIL_INSTRUCTIONS[detail_name],
            sampler_prompt_style=sampler_prompt_style,
            prompt_suffix=prompt_suffix,
            unified_instruction_ui=True,
            tile_colors=tile_colors,
        )


class SmartTileJobDirector:
    CATEGORY = "Smart Upscaler/Prompting"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "IMAGE")
    RETURN_NAMES = (
        "tile_images",
        "tile_instructions",
        "tile_references",
        "tile_seeds",
        "caption_images",
    )
    OUTPUT_IS_LIST = (True, True, True, True, True)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscaled_tiles": ("IMAGE",),
                "tile_metadata_json": ("STRING", {"forceInput": True}),
                "global_context": ("STRING", {"forceInput": True, "lazy": True}),
                "prompt_system": ("SMART_PROMPT_SYSTEM", {"forceInput": True}),
                "selection_mode": (["single_tile", "all_tiles"], {"default": "single_tile"}),
                "tile_number": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4096, "step": 1},
                ),
                "base_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "seed_mode": (["increment", "fixed", "hashed"], {"default": "increment"}),
            },
        }

    def check_lazy_status(
        self,
        upscaled_tiles,
        tile_metadata_json,
        global_context,
        prompt_system,
        selection_mode,
        tile_number,
        base_seed,
        seed_mode,
        full_image=None,
    ):
        strategy = str(_prompt_system_value(prompt_system, "prompt_strategy", "task_directed"))
        if strategy != "direct_user" and global_context is None:
            return ["global_context"]
        return []

    def build(
        self,
        upscaled_tiles,
        tile_metadata_json,
        global_context,
        prompt_system,
        selection_mode,
        tile_number,
        base_seed,
        seed_mode,
        full_image=None,
    ):
        strategy = str(_prompt_system_value(prompt_system, "prompt_strategy", "task_directed"))
        context = None if strategy == "direct_user" else global_context
        images, instructions, references = _build_all_tile_jobs(
            upscaled_tiles, tile_metadata_json, context, prompt_system
        )
        # The local caption model receives the exact tile only. A mostly empty
        # locator panel caused literal models to describe black placeholders,
        # especially for tiles at the right edge. Global context is already
        # supplied as text by the separate whole-image caption stage.
        # Caption only the real source-derived pixels. Sampler tiles remain padded
        # to one common shape, while this crop prevents padding from dominating
        # small edge tiles and reduces vision-token/VRAM cost.
        vision_images = []
        for image, reference_text in zip(images, references):
            reference = json.loads(reference_text)
            valid = reference.get("valid_output_size", {})
            valid_width = min(int(image.shape[2]), max(1, int(valid.get("width", image.shape[2]))))
            valid_height = min(
                int(image.shape[1]), max(1, int(valid.get("height", image.shape[1])))
            )
            vision_images.append(image[:, :valid_height, :valid_width, :])
        if selection_mode == "single_tile":
            selected_index = int(tile_number) - 1
            if not 0 <= selected_index < len(images):
                raise ValueError(
                    f"Tile number {tile_number} is outside the available range 1-{len(images)}."
                )
            images = [images[selected_index]]
            instructions = [instructions[selected_index]]
            references = [references[selected_index]]
            vision_images = [vision_images[selected_index]]

        seeds = [
            SmartTileSeed().derive(reference, int(base_seed), seed_mode)[0]
            for reference in references
        ]
        return images, instructions, references, seeds, vision_images
