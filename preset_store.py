import json
import os
from pathlib import Path
import tempfile
import threading


_PRESET_LOCK = threading.RLock()
_PRESET_KINDS = ("complete", "legacy")
_MAX_NAME_LENGTH = 100
_MAX_PRESET_BYTES = 256 * 1024


def _preset_path():
    return Path(__file__).resolve().parent / "presets" / "user_prompt_presets.json"


def _builtin_preset_path():
    return Path(__file__).resolve().parent / "presets" / "task_presets.json"


def _pack_directory():
    """Drop-in preset packs. Anything here is merged into the Load Preset list.

    Add-on packs live here rather than being merged into task_presets.json so a
    customer can drop a file in, and so updating the node pack never overwrites
    what they bought. The directory is git-ignored: nothing in it ships.
    """
    return Path(__file__).resolve().parent / "presets" / "packs"


def _read_preset_file(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    presets = payload.get("presets") if isinstance(payload, dict) else None
    if not isinstance(presets, dict):
        return {}
    return {
        str(name): value
        for name, value in presets.items()
        if str(name).strip() and isinstance(value, dict)
    }


def load_builtin_task_presets():
    """The shipped presets, plus any drop-in packs from presets/packs/*.json.

    Packs are read in filename order after the shipped set, so a pack may
    deliberately replace a shipped preset by reusing its name.
    """
    presets = _read_preset_file(_builtin_preset_path())
    directory = _pack_directory()
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            presets.update(_read_preset_file(path))
    return presets


def _empty_store():
    return {"version": 1, "complete": {}, "legacy": {}}


def load_prompt_presets(path=None):
    target = Path(path) if path is not None else _preset_path()
    with _PRESET_LOCK:
        if not target.exists():
            return _empty_store()
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _empty_store()
        if not isinstance(payload, dict):
            return _empty_store()
        store = _empty_store()
        for kind in _PRESET_KINDS:
            presets = payload.get(kind, {})
            if isinstance(presets, dict):
                store[kind] = {
                    str(name): value
                    for name, value in presets.items()
                    if str(name).strip() and isinstance(value, dict)
                }
        return store


def save_prompt_preset(kind, name, preset, path=None):
    kind = str(kind).strip().lower()
    if kind not in _PRESET_KINDS:
        raise ValueError("Preset kind must be complete or legacy.")
    clean_name = " ".join(str(name or "").split())
    if not clean_name:
        raise ValueError("Preset name cannot be blank.")
    if len(clean_name) > _MAX_NAME_LENGTH:
        raise ValueError(f"Preset name cannot exceed {_MAX_NAME_LENGTH} characters.")
    if not isinstance(preset, dict):
        raise ValueError("Preset data must be a JSON object.")
    if len(json.dumps(preset, ensure_ascii=False).encode("utf-8")) > _MAX_PRESET_BYTES:
        raise ValueError("Preset data is too large.")

    target = Path(path) if path is not None else _preset_path()
    with _PRESET_LOCK:
        store = load_prompt_presets(target)
        store[kind][clean_name] = preset
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.stem}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                json.dump(store, temporary, ensure_ascii=False, indent=2)
                temporary.write("\n")
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, target)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)
    return clean_name, target


def _register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return

    @PromptServer.instance.routes.get("/smart_upscaler/prompt_presets")
    async def get_prompt_presets(_request):
        return web.json_response(
            {**load_prompt_presets(), "builtin": load_builtin_task_presets()}
        )

    @PromptServer.instance.routes.post("/smart_upscaler/prompt_presets")
    async def post_prompt_preset(request):
        try:
            body = await request.json()
            name, target = save_prompt_preset(
                body.get("kind", "complete"),
                body.get("name", ""),
                body.get("preset", {}),
            )
        except (ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except OSError as exc:
            return web.json_response(
                {"error": f"Could not write the preset file: {exc}"}, status=500
            )
        return web.json_response(
            {"ok": True, "name": name, "path": str(target), "store": load_prompt_presets(target)}
        )


_register_routes()

