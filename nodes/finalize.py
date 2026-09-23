import json

import torch

from .blending import SmartTileBlender
from .fidelity import SmartTileFidelityColorMatch
from .processing import (
    SmartTileMergePartialBatch,
    _flatten_image_values,
    _single_metadata,
)


REFERENCE_BEHAVIORS = (
    "Stay close to the original photo",
    "Keep the new look (only fix warped shapes)",
)

REFERENCE_BEHAVIOR_ALIASES = {
    REFERENCE_BEHAVIORS[0]: "appearance_and_structure",
    REFERENCE_BEHAVIORS[1]: "structure_only",
    # Older graphs and API calls keep working with the previous wording.
    "Match original appearance + structure (normal upscale)": "appearance_and_structure",
    "Keep generated appearance; guide structure only": "structure_only",
    "appearance_and_structure": "appearance_and_structure",
    "structure_only": "structure_only",
}

CONSISTENCY_MODES = (
    "Shift the whole tile evenly",
    "Fade the fix toward the edge that disagrees (best for seams)",
)

CONSISTENCY_MODE_ALIASES = {
    CONSISTENCY_MODES[0]: "even",
    CONSISTENCY_MODES[1]: "gradient",
    # Older graphs keep working with the previous wording.
    "Even shift per tile": "even",
    "Smooth gradient (fixes side-to-side mismatch)": "gradient",
    "even": "even",
    "gradient": "gradient",
}

DETAIL_FREEDOM = (
    "Keep less new detail",
    "Balanced",
    "Keep more new detail",
)

# Quick-correction presets: each maps to exact dial values so results are
# reproducible. "Manual" leaves the dials in charge (previous behavior).
FINISH_PRESETS = {
    "Manual - I set the dials myself": None,
    # A faithful upscale that also hides its joins. Every other faithful preset
    # left the seam dials weak, so the one combination people actually want on a
    # correct photograph had to be built by hand.
    "Photo upscale, seams hidden (start here)": {
        "reference_mode": "Stay close to the original photo",
        "structure_preservation": 50,
        "detail_support": "Keep more new detail",
        "cross_tile_consistency": 60,
        "consistency_mode": "Fade the fix toward the edge that disagrees (best for seams)",
    },
    "Photo upscale, hug the original": {
        "reference_mode": "Stay close to the original photo",
        "structure_preservation": 55,
        "detail_support": "Balanced",
        "cross_tile_consistency": 35,
        "consistency_mode": "Shift the whole tile evenly",
    },
    "I changed the look (style, time of day)": {
        "reference_mode": "Keep the new look (only fix warped shapes)",
        "structure_preservation": 20,
        "detail_support": "Balanced",
        "cross_tile_consistency": 35,
        "consistency_mode": "Shift the whole tile evenly",
    },
    "I changed the look, seams hidden": {
        "reference_mode": "Keep the new look (only fix warped shapes)",
        "structure_preservation": 20,
        "detail_support": "Balanced",
        "cross_tile_consistency": 60,
        "consistency_mode": "Fade the fix toward the edge that disagrees (best for seams)",
    },
    "Let the model add the most detail": {
        "reference_mode": "Keep the new look (only fix warped shapes)",
        "structure_preservation": 10,
        "detail_support": "Keep more new detail",
        "cross_tile_consistency": 35,
        "consistency_mode": "Shift the whole tile evenly",
    },
}

DETAIL_FREEDOM_ALIASES = {
    DETAIL_FREEDOM[0]: "conservative",
    DETAIL_FREEDOM[1]: "balanced",
    DETAIL_FREEDOM[2]: "permissive",
    # Older graphs keep working with the previous wording.
    "Conservative (less generated detail)": "conservative",
    "Permissive (more generated detail)": "permissive",
    "conservative": "conservative",
    "balanced": "balanced",
    "permissive": "permissive",
}


def _first(values, default=None):
    flattened = _flatten_image_values(values)
    return flattened[0] if flattened else default


def _output_rect(tile_record, scale):
    x0 = round(float(tile_record["x"]) * scale)
    y0 = round(float(tile_record["y"]) * scale)
    x1 = max(x0 + 1, round((float(tile_record["x"]) + float(tile_record["width"])) * scale))
    y1 = max(y0 + 1, round((float(tile_record["y"]) + float(tile_record["height"])) * scale))
    return x0, y0, x1, y1


def _cross_tile_overlap_consistency(tiles, metadata, strength, mode="even"):
    """Align low-frequency RGB offsets between neighbors using shared overlap pixels.

    ``mode="even"`` applies one constant offset per tile (the long-standing
    behavior). ``mode="gradient"`` additionally ramps a smooth per-edge
    correction across each tile, so a tile that only disagrees with its
    neighbor on one side is corrected on that side instead of everywhere.
    """

    amount = max(0.0, min(1.0, float(strength) / 100.0))
    if amount <= 0.0 or tiles.shape[0] < 2:
        return tiles
    records = sorted(metadata.get("tiles", []), key=lambda item: int(item["tile_index"]))
    if len(records) != tiles.shape[0]:
        return tiles
    scale = float(metadata.get("scale_factor", 1.0))
    result = tiles.clone()

    source_groups = {}
    for record in records:
        source_groups.setdefault(int(record.get("source_index", 0)), []).append(record)

    for group in source_groups.values():
        if len(group) < 2:
            continue
        grid = {(int(item["row"]), int(item["column"])): item for item in group}
        group_indexes = [int(item["tile_index"]) for item in group]
        local_index = {tile_index: index for index, tile_index in enumerate(group_indexes)}
        rows = []
        differences = []
        pairs = []

        for record in group:
            row = int(record["row"])
            column = int(record["column"])
            for neighbor_position in ((row, column + 1), (row + 1, column)):
                neighbor = grid.get(neighbor_position)
                if neighbor is None:
                    continue
                first_index = int(record["tile_index"])
                second_index = int(neighbor["tile_index"])
                ax0, ay0, ax1, ay1 = _output_rect(record, scale)
                bx0, by0, bx1, by1 = _output_rect(neighbor, scale)
                ox0, oy0 = max(ax0, bx0), max(ay0, by0)
                ox1, oy1 = min(ax1, bx1), min(ay1, by1)
                if ox1 <= ox0 or oy1 <= oy0:
                    continue

                first_patch = tiles[
                    first_index,
                    oy0 - ay0 : oy1 - ay0,
                    ox0 - ax0 : ox1 - ax0,
                    :,
                ]
                second_patch = tiles[
                    second_index,
                    oy0 - by0 : oy1 - by0,
                    ox0 - bx0 : ox1 - bx0,
                    :,
                ]
                if first_patch.numel() == 0 or second_patch.numel() == 0:
                    continue
                first_mean = first_patch.float().mean(dim=(0, 1))
                second_mean = second_patch.float().mean(dim=(0, 1))
                equation = torch.zeros(
                    (len(group_indexes),), device=tiles.device, dtype=torch.float32
                )
                equation[local_index[first_index]] = -1.0
                equation[local_index[second_index]] = 1.0
                rows.append(equation)
                differences.append(first_mean - second_mean)
                pairs.append(
                    (
                        first_index,
                        second_index,
                        "horizontal" if neighbor_position[1] == column + 1 else "vertical",
                        first_mean - second_mean,
                    )
                )

        if not rows:
            continue

        if mode == "gradient":
            # Gradient mode: correct each seam to its midpoint with a smooth ramp
            # across each tile. A tile whose color drifts toward one side gets
            # counter-corrected exactly there, while a tile that already agrees
            # with its neighbors barely moves — seams close without dragging
            # correct tiles toward a drifting one, and without touching the
            # source. This is per-tile, individual, gradient color correction.
            height, width = tiles.shape[1], tiles.shape[2]
            ramp_x = torch.linspace(0.0, 1.0, width, device=tiles.device).view(1, width, 1)
            ramp_y = torch.linspace(0.0, 1.0, height, device=tiles.device).view(height, 1, 1)
            zero = torch.zeros((tiles.shape[-1],), device=tiles.device, dtype=torch.float32)
            edge_corrections = {}
            for first_index, second_index, orientation, diff in pairs:
                half = (diff / 2.0).clamp(-0.12, 0.12)
                first_sides = edge_corrections.setdefault(first_index, {})
                second_sides = edge_corrections.setdefault(second_index, {})
                if orientation == "horizontal":
                    first_sides["right"] = first_sides.get("right", zero) - half
                    second_sides["left"] = second_sides.get("left", zero) + half
                else:
                    first_sides["bottom"] = first_sides.get("bottom", zero) - half
                    second_sides["top"] = second_sides.get("top", zero) + half
            for tile_index, sides in edge_corrections.items():
                left = sides.get("left", zero).view(1, 1, -1)
                right = sides.get("right", zero).view(1, 1, -1)
                top = sides.get("top", zero).view(1, 1, -1)
                bottom = sides.get("bottom", zero).view(1, 1, -1)
                field = (
                    left * (1.0 - ramp_x)
                    + right * ramp_x
                    + top * (1.0 - ramp_y)
                    + bottom * ramp_y
                ).clamp(-0.12, 0.12).to(dtype=tiles.dtype)
                result[tile_index] = (
                    tiles[tile_index] + field * amount
                ).clamp(0.0, 1.0)
            continue

        # Even mode: one constant offset per tile. A zero-sum anchor preserves
        # the batch's overall transformed color while solving only relative
        # disagreement between neighboring overlaps.
        rows.append(
            torch.ones(
                (len(group_indexes),), device=tiles.device, dtype=torch.float32
            )
            / max(1.0, len(group_indexes) ** 0.5)
        )
        differences.append(torch.zeros((tiles.shape[-1],), device=tiles.device))
        matrix = torch.stack(rows, dim=0)
        targets = torch.stack(differences, dim=0)
        offsets = torch.linalg.lstsq(matrix, targets).solution.clamp(-0.08, 0.08)
        for position, tile_index in enumerate(group_indexes):
            offset = offsets[position].to(device=tiles.device, dtype=tiles.dtype)
            result[tile_index] = (tiles[tile_index] + offset * amount).clamp(0.0, 1.0)
    return result


class SmartTileFinalizer:
    CATEGORY = "Smart Upscaler/Processing"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("corrected_tiles", "final_image")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "finalize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_images": ("IMAGE", {"forceInput": True}),
                "tile_references": ("STRING", {"forceInput": True}),
                "source_tiles": ("IMAGE", {"forceInput": True}),
                "tile_metadata_json": ("STRING", {"forceInput": True}),
                "blend_masks": ("MASK", {"forceInput": True}),
                "reference_mode": (
                    list(REFERENCE_BEHAVIORS),
                    {
                        "default": REFERENCE_BEHAVIORS[0],
                        "advanced": True,
                        "label": "1. Follow the original, or keep the new look?",
                        "tooltip": "The most important setting here.\n\nStay close to the original: every tile is pulled back toward the source's own colour and brightness. Use this whenever the source was already a correct picture - it is also the strongest cure for visible tiles, because it gives them all the same starting tone.\n\nKeep the new look: each tile keeps the colour and brightness the model gave it, and only bent shapes are corrected. Use this when you deliberately changed the picture - a style, a time of day, a repair of a broken source.",
                    },
                ),
                "structure_preservation": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                        "advanced": True,
                        "label": "2. How closely to follow it",
                        "tooltip": "How strongly setting 1 is applied. Around 50 is a normal photo upscale. Below about 20 it barely does anything, so a low number here undoes setting 1 - raise it if the result drifts from your original.",
                    },
                ),
                "detail_support": (
                    list(DETAIL_FREEDOM),
                    {
                        "default": "Keep more new detail",
                        "advanced": True,
                        "label": "3. How much of the model's new detail to keep",
                        "tooltip": "Settings 1 and 2 pull the picture back toward the source. This decides how much of the fine detail the model just generated survives that pull.\n\nKeep more new detail is the right choice when the model genuinely improved the texture and you only want the source to fix colour and shape.",
                    },
                ),
                "fallback_method": (
                    ["bilinear", "bicubic", "area", "nearest-exact"],
                    {
                        "default": "bilinear",
                        "advanced": True,
                        "label": "(rarely used) Filler for tiles you have not rendered",
                        "tooltip": "Only used during a one-tile test, to fill in the tiles you have not rendered yet so you can see the whole picture. Has no effect on a normal full run. Leave it alone.",
                    },
                ),
                "cross_tile_consistency": (
                    "INT",
                    {
                        "default": 60,
                        "min": 0,
                        "max": 100,
                        "step": 5,
                        "display": "slider",
                        "advanced": True,
                        "label": "4. Hide the joins between tiles",
                        "tooltip": "Evens out colour and brightness differences between neighbouring tiles, using the strip of picture they share. 0 is off; 60 is strong.\n\nThis fixes tiles that came out different SHADES. It cannot fix tiles that generated different TEXTURE - if you can see a grid in flat areas like water or sky, that comes from the prompt, so check the prompt log first.",
                    },
                ),
            },
            "optional": {
                "consistency_mode": (
                    list(CONSISTENCY_MODES),
                    {
                        "default": CONSISTENCY_MODES[1],
                        "advanced": True,
                        "label": "5. How to hide them",
                        "tooltip": "Shift the whole tile evenly: one correction for the whole tile. Fine when a tile is uniformly off.\n\nFade toward the edge that disagrees: the correction is strongest at the join and fades away across the tile, so a tile that only mismatches on one side is fixed there instead of everywhere. This is the better choice whenever you can actually see the joins.",
                    },
                ),
                "finish_preset": (
                    list(FINISH_PRESETS),
                    {
                        "default": "Photo upscale, seams hidden (start here)",
                        "label": "Quick Preset - sets 1 to 5 for you",
                        "tooltip": "The only setting most pictures need. It fills in dials 1 to 5 in Advanced, then stays on screen as a label so you can see what you chose and adjust from there.\n\nPhoto upscale, seams hidden: start here for a normal upscale of a picture that was already correct.\n\nUse the 'I changed the look' presets when you asked for a style, a time of day, or a repair, so your new look is not pulled back to the original.",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, reference_mode=None, finish_preset=None, consistency_mode=None):
        # Accept stored values from older graphs (previous label wording); the
        # alias maps and .get() fallbacks resolve anything unknown safely.
        return True

    def finalize(
        self,
        processed_images,
        tile_references,
        source_tiles,
        tile_metadata_json,
        blend_masks,
        reference_mode,
        structure_preservation,
        detail_support,
        fallback_method,
        cross_tile_consistency=35,
        consistency_mode=CONSISTENCY_MODES[0],
        finish_preset="Manual - I set the dials myself",
    ):
        # The preset dropdown writes values into the visible dials (see
        # web/finishing_presets.js) and stays selected as a label; the dials are
        # always the source of truth here.
        metadata = _single_metadata(tile_metadata_json)
        metadata_text = str(_first(tile_metadata_json, ""))
        processed = _flatten_image_values(processed_images)
        references = [str(value) for value in _flatten_image_values(tile_references)]
        source_batch = _first(source_tiles)
        masks = _first(blend_masks)
        if not isinstance(source_batch, torch.Tensor) or source_batch.ndim != 4:
            raise ValueError("Source tiles must come from a Smart tile planner.")
        if not isinstance(masks, torch.Tensor):
            raise ValueError("Blend masks must come from the same Smart tile planner.")
        if len(processed) != len(references):
            raise ValueError("Processed tile images and references must have matching counts.")

        mode_value = str(_first(reference_mode, REFERENCE_BEHAVIORS[0]))
        mode = REFERENCE_BEHAVIOR_ALIASES.get(mode_value, "appearance_and_structure")
        structure = int(_first(structure_preservation, 55))
        support_value = str(_first(detail_support, "Balanced"))
        support = DETAIL_FREEDOM_ALIASES.get(support_value, "balanced")
        fallback = str(_first(fallback_method, "bilinear"))
        consistency = int(_first(cross_tile_consistency, 35))
        mode_choice = str(_first(consistency_mode, CONSISTENCY_MODES[0]))
        seam_mode = CONSISTENCY_MODE_ALIASES.get(
            mode_choice,
            "gradient" if "gradient" in mode_choice.lower() or "fade" in mode_choice.lower() else "even",
        )

        corrected = []
        for processed_image, reference_text in zip(processed, references):
            try:
                reference = json.loads(reference_text)
            except json.JSONDecodeError as exc:
                raise ValueError("Tile references must come from Smart Tile Job Director.") from exc
            tile_index = int(reference.get("tile_index", -1))
            if not 0 <= tile_index < source_batch.shape[0]:
                raise ValueError(f"Processed tile has invalid index {tile_index}.")
            source_tile = source_batch[tile_index : tile_index + 1]
            corrected_tile = SmartTileFidelityColorMatch().apply(
                processed_image,
                source_tile,
                structure_preservation=structure,
                detail_support=support,
                color_match_method="none",
                color_match_strength=0,
                reference_mode=mode,
            )[0]
            corrected.append(corrected_tile)

        merged = SmartTileMergePartialBatch().merge(
            corrected,
            references,
            [source_batch],
            [metadata_text],
            [fallback],
        )[0]
        expected_count = len(metadata.get("tiles", []))
        if merged.shape[0] != expected_count:
            raise ValueError("Finalizer did not reconstruct the complete planned tile batch.")
        if len(corrected) == expected_count and consistency > 0:
            merged = _cross_tile_overlap_consistency(
                merged, metadata, consistency, seam_mode
            )
            corrected = []
            for reference_text in references:
                reference = json.loads(reference_text)
                tile_index = int(reference["tile_index"])
                corrected.append(merged[tile_index : tile_index + 1])
        final_image = SmartTileBlender().blend(
            merged,
            metadata_text,
            masks,
            "metadata_scale_factor",
        )[0]
        return corrected, final_image
