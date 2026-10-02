"""Stitch Tiles - Clean: put the generated tiles back together without ever
mixing the original's fine detail into them.

The older Stitch Tiles Into One Picture node blends source detail with the
generated detail to guard shapes, which can leave a speckled look. Here the
source only ever contributes broad colour (or nothing), so every fine detail in
the result is exactly what the generator drew.
"""

import json

import torch

from .blending import SmartTileBlender
from .fidelity import _automatic_method, _color_match, _prepare_source
from .finalize import _cross_tile_overlap_consistency, _first
from .processing import (
    SmartTileMergePartialBatch,
    _flatten_image_values,
    _rgb,
    _single_metadata,
)


COLOR_CHOICES = (
    "Automatic - follow the Prompt Director (recommended)",
    "Match the original's colors (photo upscale)",
    "Keep the new look, even out the tiles (style, time of day)",
    "Leave the tiles exactly as drawn",
)

JOIN_CHOICES = (
    "Soft - fade across the overlap (recommended)",
    "Hard cut - stitch only, no blending",
)

# Evening out a changed look: one colour offset per tile, solved for all tiles
# at once from the overlaps they share. Measured on real Klein content with tile
# tints, this cut neighbour mismatch from 10.9 to 1.8 (0-255); fading each join
# separately ("gradient") only reached 4.0 and added nothing on top.
_EVEN_OUT_STRENGTH = 100


def _color_plan(choice, prompt_system):
    if choice == COLOR_CHOICES[0]:
        return "even" if _automatic_method(prompt_system) == "none" else "original"
    return {
        COLOR_CHOICES[2]: "even",
        COLOR_CHOICES[3]: "none",
    }.get(choice, "original")


def _hard_cut_masks(metadata, planner_masks):
    """Weight 1 inside each tile's own core rectangle, 0 elsewhere.

    The planner's cores partition the picture, and both edges of every core are
    rounded from absolute positions, so neighbours meet without a gap or a
    double-covered pixel at any scale.
    """
    scale = float(metadata.get("scale_factor", 1.0))
    masks = torch.zeros_like(planner_masks)
    height, width = masks.shape[1:3]
    for record in metadata["tiles"]:
        x0 = round(float(record["x"]) * scale)
        y0 = round(float(record["y"]) * scale)
        core_x = float(record.get("core_x", record["x"]))
        core_y = float(record.get("core_y", record["y"]))
        core_width = float(record.get("core_width", record["width"]))
        core_height = float(record.get("core_height", record["height"]))
        left = max(0, round(core_x * scale) - x0)
        top = max(0, round(core_y * scale) - y0)
        right = min(width, round((core_x + core_width) * scale) - x0)
        bottom = min(height, round((core_y + core_height) * scale) - y0)
        masks[int(record["tile_index"]), top:bottom, left:right] = 1.0
    return masks


class SmartTileStitchClean:
    CATEGORY = "Smart Upscaler/Processing"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("corrected_tiles", "final_image")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "stitch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_images": ("IMAGE", {"forceInput": True}),
                "tile_references": ("STRING", {"forceInput": True}),
                "source_tiles": ("IMAGE", {"forceInput": True}),
                "tile_metadata_json": ("STRING", {"forceInput": True}),
                "blend_masks": ("MASK", {"forceInput": True}),
                "colors": (
                    list(COLOR_CHOICES),
                    {
                        "default": COLOR_CHOICES[0],
                        "tooltip": "Automatic: matches the original's colors for an upscale, and keeps your new look when the Prompt Director's task is a style or time-of-day change (connect its prompt_system; without it, Automatic matches the original).\n\nMatch the original's colors: every tile takes the original's broad color, so neighbours agree. Fine detail always stays exactly as the model drew it.\n\nKeep the new look, even out the tiles: tiles are only nudged to agree with each other at the joins.\n\nLeave the tiles exactly as drawn: no color change at all.",
                    },
                ),
                "joins": (
                    list(JOIN_CHOICES),
                    {
                        "default": JOIN_CHOICES[0],
                        "tooltip": "Soft: neighbouring tiles fade into each other across their overlap, as set by the Tile Planner's overlap and feather.\n\nHard cut: each part of the picture comes from exactly one tile, with no blending at all. Useful to see exactly what each tile drew; joins can show as lines.",
                    },
                ),
            },
            "optional": {
                "prompt_system": ("SMART_PROMPT_SYSTEM", {"forceInput": True}),
            },
        }

    def stitch(
        self,
        processed_images,
        tile_references,
        source_tiles,
        tile_metadata_json,
        blend_masks,
        colors,
        joins,
        prompt_system=None,
    ):
        metadata = _single_metadata(tile_metadata_json)
        metadata_text = str(_first(tile_metadata_json, ""))
        processed = [_rgb(image) for image in _flatten_image_values(processed_images)]
        references = [str(value) for value in _flatten_image_values(tile_references)]
        source_batch = _first(source_tiles)
        masks = _first(blend_masks)
        if not isinstance(source_batch, torch.Tensor) or source_batch.ndim != 4:
            raise ValueError("Source tiles must come from a Smart tile planner.")
        if not isinstance(masks, torch.Tensor):
            raise ValueError("Blend masks must come from the same Smart tile planner.")
        if len(processed) != len(references):
            raise ValueError("Processed tile images and references must have matching counts.")

        plan = _color_plan(str(_first(colors, COLOR_CHOICES[0])), _first(prompt_system))
        corrected = []
        for image, reference_text in zip(processed, references):
            try:
                tile_index = int(json.loads(reference_text).get("tile_index", -1))
            except json.JSONDecodeError as exc:
                raise ValueError("Tile references must come from Smart Tile Job Director.") from exc
            if not 0 <= tile_index < source_batch.shape[0]:
                raise ValueError(f"Processed tile has invalid index {tile_index}.")
            if plan == "original":
                source = _prepare_source(image, source_batch[tile_index : tile_index + 1])
                image = _color_match(image, source, "original_colors").clamp(0.0, 1.0)
            corrected.append(image)

        # A one-tile test fills the tiles it did not render from the source.
        merged = SmartTileMergePartialBatch().merge(
            corrected, references, [source_batch], [metadata_text], ["bilinear"]
        )[0]
        if merged.shape[0] != len(metadata.get("tiles", [])):
            raise ValueError("Stitching did not reconstruct the complete planned tile batch.")
        if plan == "even" and len(corrected) == merged.shape[0]:
            merged = _cross_tile_overlap_consistency(
                merged, metadata, _EVEN_OUT_STRENGTH, "even"
            )
            corrected = [
                merged[int(json.loads(text)["tile_index"])].unsqueeze(0) for text in references
            ]

        join = str(_first(joins, JOIN_CHOICES[0]))
        weights = _hard_cut_masks(metadata, masks) if join == JOIN_CHOICES[1] else masks
        final_image = SmartTileBlender().blend(
            merged, metadata_text, weights, "metadata_scale_factor"
        )[0]
        return corrected, final_image
