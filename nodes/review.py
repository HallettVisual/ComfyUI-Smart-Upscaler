import json
from pathlib import Path
import uuid

import numpy as np
import torch
from PIL import Image


def _flatten(values):
    flattened = []
    pending = list(values if isinstance(values, list) else [values])
    while pending:
        value = pending.pop(0)
        if isinstance(value, list):
            pending[0:0] = value
        else:
            flattened.append(value)
    return flattened


def _first(values, default=None):
    flattened = _flatten(values)
    return flattened[0] if flattened else default




def _save_inspector_image(image, directory, prefix):
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Each inspector image must contain exactly one ComfyUI image.")
    filename = f"{prefix}_{uuid.uuid4().hex[:12]}.png"
    image_array = (
        image[0].detach().float().clamp(0.0, 1.0).cpu().numpy() * 255.0
    ).round().astype(np.uint8)
    Image.fromarray(image_array, mode="RGB").save(directory / filename, compress_level=1)
    return {
        "filename": filename,
        "subfolder": "smart_upscaler",
        "type": "temp",
    }


class SmartTileInspector:
    CATEGORY = "Smart Upscaler/Review"
    RETURN_TYPES = ()
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "inspect"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_images": ("IMAGE", {"forceInput": True}),
                "generated_images": ("IMAGE", {"forceInput": True}),
                "prompts": ("STRING", {"forceInput": True}),
                "tile_references": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def inspect(
        self,
        source_images,
        generated_images,
        prompts,
        tile_references,
        unique_id=None,
    ):
        import folder_paths

        sources = _flatten(source_images)
        generated = _flatten(generated_images)
        prompt_values = [str(value) for value in _flatten(prompts)]
        references = _flatten(tile_references)
        if not (len(sources) == len(generated) == len(prompt_values) == len(references)):
            raise ValueError(
                "Inspector source images, generated images, prompts, and references must match."
            )

        directory = Path(folder_paths.get_temp_directory()) / "smart_upscaler"
        directory.mkdir(parents=True, exist_ok=True)
        node_prefix = f"tile_review_{str(_first(unique_id, 'node')).replace(':', '_')}"
        records = []
        for source_image, generated_image, prompt, reference_json in zip(
            sources,
            generated,
            prompt_values,
            references,
        ):
            try:
                reference = json.loads(reference_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("Tile references must come from a Smart tile prompt batch.") from exc
            tile_index = int(reference.get("tile_index", len(records)))
            tile_id = str(reference.get("tile_id", f"T{tile_index + 1:03d}"))
            records.append(
                {
                    "tile_id": tile_id,
                    "tile_index": tile_index,
                    "position": reference.get("position", "unknown"),
                    "row": int(reference.get("row", 0)) + 1,
                    "column": int(reference.get("column", 0)) + 1,
                    "prompt": prompt.strip() or "[empty prompt]",
                    "source": _save_inspector_image(
                        source_image,
                        directory,
                        f"{node_prefix}_{tile_id}_source",
                    ),
                    "generated": _save_inspector_image(
                        generated_image,
                        directory,
                        f"{node_prefix}_{tile_id}_generated",
                    ),
                }
            )
        return {"ui": {"smart_tile_inspector": records}}
