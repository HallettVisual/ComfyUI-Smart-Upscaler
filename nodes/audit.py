import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import tempfile
import uuid

def _flatten_values(values):
    flattened = []
    pending = list(values if isinstance(values, (list, tuple)) else [values])
    while pending:
        value = pending.pop(0)
        if isinstance(value, (list, tuple)):
            pending[0:0] = value
        else:
            flattened.append(value)
    return flattened


def _first(values, default=None):
    flattened = _flatten_values(values)
    return flattened[0] if flattened else default


def _as_text_list(values):
    return [str(value) for value in _flatten_values(values)]


def _aligned(values, count, default=""):
    items = _as_text_list(values)
    if len(items) == count:
        return items
    if len(items) == 1 and count > 1:
        return items * count
    if not items:
        return [default] * count
    raise ValueError(f"Audit input has {len(items)} items; expected {count}.")


def _parse_object(value):
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {"raw": str(value)}
    return parsed if isinstance(parsed, dict) else {"raw": str(value)}


def _log_directory():
    try:
        import folder_paths

        root = Path(folder_paths.get_output_directory())
    except ImportError:
        root = Path(tempfile.gettempdir())
    directory = root / "Smart-Upscaler" / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_label(value):
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-._")
    return label[:64] or "tile-prompts"


def _source_inputs(prompt_graph):
    prompt = _first(prompt_graph, {})
    if not isinstance(prompt, dict):
        return []
    sources = []
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {})
        if class_type == "LoadImage" and isinstance(inputs, dict):
            sources.append(
                {
                    "node_id": str(node_id),
                    "class_type": class_type,
                    "image": inputs.get("image"),
                }
            )
    return sources


def _atomic_write(path, text):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _scene_summary_lines(global_context_text):
    """One-line scene + surface summary parsed from the master brief.

    The full master JSON stays in the JSON log for debugging; the readable log only needs
    a compact human summary, not the raw schema.
    """
    text = str(global_context_text or "").strip()
    if text.startswith("```"):
        body = text.splitlines()
        if body and body[0].startswith("```"):
            body = body[1:]
        if body and body[-1].startswith("```"):
            body = body[:-1]
        text = "\n".join(body).strip()
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    out = []
    scene = str(data.get("scene_type", "")).strip()
    geo = str(data.get("geographic_context", "")).strip()
    if scene or geo:
        out.append("Scene: " + " — ".join(part for part in (scene, geo) if part))
    surfaces = data.get("surface_map")
    if isinstance(surfaces, list) and surfaces:
        parts = []
        for surface in surfaces:
            if not isinstance(surface, dict):
                continue
            name = str(surface.get("identity") or surface.get("id") or "").strip()
            locations = surface.get("locations")
            where = ", ".join(map(str, locations)) if isinstance(locations, list) else ""
            if name:
                parts.append(f"{name} ({where})" if where else name)
        if parts:
            out.append("Continuous surfaces: " + "; ".join(parts))
    materials = data.get("material_map")
    if isinstance(materials, list) and materials:
        parts = []
        for entry in materials:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("identity") or entry.get("id") or "").strip()
            locations = entry.get("locations")
            where = ", ".join(map(str, locations)) if isinstance(locations, list) else ""
            if name:
                parts.append(f"{name} ({where})" if where else name)
        if parts:
            out.append("Shared materials: " + "; ".join(parts))
    objects = data.get("object_map")
    if isinstance(objects, list) and objects:
        parts = []
        for entry in objects:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("identity") or entry.get("id") or "").strip()
            if not name:
                continue
            part_map = entry.get("parts")
            if isinstance(part_map, dict) and part_map:
                where = "; ".join(
                    f"{label}: {text}" for label, text in part_map.items()
                )
            else:
                locations = entry.get("locations")
                where = (
                    ", ".join(map(str, locations))
                    if isinstance(locations, list)
                    else ""
                )
            parts.append(f"{name} ({where})" if where else name)
        if parts:
            out.append("Main subject / objects: " + "; ".join(parts))
    return out


def _tile_needed_recovery(tile):
    """True when this tile did not read cleanly on the first attempt."""
    status = str((tile or {}).get("cache_status", "")).upper()
    return any(
        marker in status
        for marker in ("CONSERVATIVE-FALLBACK", "PLAIN-RECOVERY", "SCHEMA-GUARD")
    )


def _tile_health_lines(tiles):
    """A run summary, because failures are invisible in a 30-tile wall of text.

    On a 9-tile image a bad tile is obvious. At 30+ tiles it is five lines buried
    in three pages, and the user has to find them by eye. Anything that did not
    read cleanly is counted here and flagged inline below.
    """
    if not tiles:
        return []
    recovered = [tile["tile_id"] for tile in tiles if _tile_needed_recovery(tile)]
    retried = [
        tile["tile_id"]
        for tile in tiles
        if "RETRY" in str(tile.get("cache_status", "")).upper()
        and tile["tile_id"] not in recovered
    ]
    lines = [f"Tiles: {len(tiles)} | read cleanly: {len(tiles) - len(recovered)}"]
    if retried:
        lines.append(f"Re-asked once, then fine: {', '.join(retried)}")
    if recovered:
        lines.append(
            f"NEEDS ATTENTION - could not be read, described from scene context: "
            f"{', '.join(recovered)}"
        )
        lines.append(
            "  Re-run to retry just these (everything else is cached), or lower "
            "Prompt Detail Level - dense tiles fail most often at Maximum."
        )
    return lines + [""]


def _readable_log(payload):
    config = payload["prompt_configuration"]
    lines = ["SMART UPSCALER PROMPTS", f"Created UTC: {payload['created_utc']}", ""]

    instruction = str(
        config.get("instructions", config.get("workflow_instruction", ""))
    ).strip()
    if instruction:
        lines.extend(["INSTRUCTIONS:", instruction, ""])
    false_detections = config.get("known_false_detections") or []
    if isinstance(false_detections, str):
        false_detections = [false_detections]
    lines.append(
        "Known false detections: "
        + (", ".join(str(item) for item in false_detections) if false_detections else "none")
    )
    user_request = str(config.get("user_request", "")).strip()
    if user_request:
        lines.append(f"User request: {user_request}")
    lines.append("")

    summary = _scene_summary_lines(payload.get("global_context", ""))
    if summary:
        lines.extend(summary + [""])

    lines.extend(_tile_health_lines(payload.get("tiles") or []))

    lines.append("TILE PROMPTS")
    for tile in payload["tiles"]:
        flag = " [NEEDS ATTENTION]" if _tile_needed_recovery(tile) else ""
        lines.append(f"{tile['tile_id']} | {tile.get('position', '')}{flag}")
        lines.append(tile["final_positive_prompt"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class SmartTilePromptAuditLog:
    CATEGORY = "Smart Upscaler/Review"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "all_prompt_text",
        "json_log_path",
        "readable_log_path",
    )
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False, False, False)
    OUTPUT_NODE = True
    FUNCTION = "save"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_instructions": ("STRING", {"forceInput": True}),
                "positive_prompts": ("STRING", {"forceInput": True}),
                "negative_prompts": ("STRING", {"forceInput": True}),
                "prompt_audits": ("STRING", {"forceInput": True}),
                "tile_references": ("STRING", {"forceInput": True}),
                "cache_status": ("STRING", {"forceInput": True}),
                "prompt_system": ("SMART_PROMPT_SYSTEM", {"forceInput": True}),
                "global_instruction": ("STRING", {"forceInput": True}),
                "global_context": (
                    "STRING",
                    {"forceInput": True, "lazy": True},
                ),
                "combined_instructions": ("STRING", {"forceInput": True}),
                "log_label": (
                    "STRING",
                    {
                        "default": "tile-prompts",
                        "label": "Log Name",
                        "tooltip": "A JSON log and readable text log are saved for every queued run.",
                    },
                ),
            },
            "hidden": {"workflow_prompt": "PROMPT"},
        }

    def check_lazy_status(
        self,
        tile_instructions,
        positive_prompts,
        negative_prompts,
        prompt_audits,
        tile_references,
        cache_status,
        prompt_system,
        global_instruction,
        global_context,
        combined_instructions,
        log_label,
        workflow_prompt=None,
    ):
        system = _first(prompt_system, {})
        strategy = system.get("prompt_strategy", "task_directed") if isinstance(system, dict) else "task_directed"
        if strategy != "direct_user" and _first(global_context) is None:
            return ["global_context"]
        return []

    def save(
        self,
        tile_instructions,
        positive_prompts,
        negative_prompts,
        prompt_audits,
        tile_references,
        cache_status,
        prompt_system,
        global_instruction,
        global_context,
        combined_instructions,
        log_label,
        workflow_prompt=None,
    ):
        references = _as_text_list(tile_references)
        count = len(references)
        if count < 1:
            raise ValueError("Tile prompt audit received no tile references.")
        instructions = _aligned(tile_instructions, count)
        positives = _aligned(positive_prompts, count)
        negatives = _aligned(negative_prompts, count)
        audits = _aligned(prompt_audits, count)
        statuses = _aligned(cache_status, count)
        system = _first(prompt_system, {})
        if not isinstance(system, dict):
            system = _parse_object(system)

        tile_records = []
        for instruction, positive, negative, audit_text, reference_text, status in zip(
            instructions, positives, negatives, audits, references, statuses
        ):
            reference = _parse_object(reference_text)
            audit = _parse_object(audit_text)
            raw_response = audit.get("raw_source_response", audit.get("raw_response", ""))
            tile_records.append(
                {
                    "tile_id": reference.get(
                        "tile_id", f"T{int(reference.get('tile_index', 0)) + 1:03d}"
                    ),
                    "tile_index": int(reference.get("tile_index", 0)),
                    "position": reference.get("position", ""),
                    "reference": reference,
                    "vlm_instruction": instruction,
                    "raw_vlm_response": str(raw_response),
                    "local_source_evidence": audit.get("local_source_evidence", ""),
                    "final_positive_prompt": positive,
                    "final_negative_prompt": negative,
                    "cache_status": status,
                    "evidence_guard": audit.get("evidence_guard", ""),
                    "resolver_audit": audit,
                }
            )

        now = datetime.now(timezone.utc)
        label = _safe_label(_first(log_label, "tile-prompts"))
        payload = {
            "schema": 1,
            "created_utc": now.isoformat(),
            "label": label,
            "source_inputs": _source_inputs(workflow_prompt),
            "prompt_configuration": system,
            "combined_instructions": str(_first(combined_instructions, "")),
            "global_instruction": str(_first(global_instruction, "")),
            "global_context": (
                "[VLM bypassed]"
                if str(system.get("prompt_strategy", "")) == "direct_user"
                else str(_first(global_context, ""))
            ),
            "tile_count": count,
            "tiles": tile_records,
        }
        directory = _log_directory()
        stem = f"{label}_{now.strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:6]}"
        json_path = directory / f"{stem}.json"
        text_path = directory / f"{stem}.txt"
        json_text = json.dumps(payload, indent=2, ensure_ascii=False)
        readable_text = _readable_log(payload)
        _atomic_write(json_path, json_text)
        _atomic_write(text_path, readable_text)
        _atomic_write(directory / "tile_prompt_audit_latest.json", json_text)
        _atomic_write(directory / "tile_prompt_audit_latest.txt", readable_text)

        return {
            "ui": {
                "prompt_audit_text": [readable_text],
                "json_log_path": [str(json_path)],
                "readable_log_path": [str(text_path)],
            },
            "result": (readable_text, str(json_path), str(text_path)),
        }
