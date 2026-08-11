import json

import torch
import torch.nn.functional as F


def _stitch_memory_guard(output_width, output_height, channels, image_batch, available_bytes=None):
    """Refuse impossible canvases with a plain answer instead of an allocator crash.

    The stitched image lives in system RAM as float32. Peak use during blending
    is roughly canvas + weights (~1.4x canvas). When that cannot fit, tell the
    user the real numbers and the largest scale that would.
    """
    canvas_bytes = float(output_width) * float(output_height) * channels * 4.0 * max(1, image_batch)
    needed = canvas_bytes * 1.6
    if available_bytes is None:
        try:
            import psutil

            available_bytes = psutil.virtual_memory().available
        except Exception:
            return
    if needed <= float(available_bytes):
        return
    gigapixels = output_width * output_height / 1e9
    feasible = (float(available_bytes) / 1.6 / (channels * 4.0 * max(1, image_batch))) ** 0.5
    raise ValueError(
        f"The requested result would be {output_width}x{output_height} pixels "
        f"({gigapixels:.2f} gigapixels, about {canvas_bytes / 1024**3:.1f} GB in memory), "
        f"but only {float(available_bytes) / 1024**3:.1f} GB of system memory is free. "
        f"Lower the scale factor so the output stays under roughly "
        f"{int(feasible)}x{int(feasible * output_height / max(1, output_width))} pixels, "
        "or free up memory and try again."
    )


class SmartTileBlender:
    CATEGORY = "Smart Upscaler/Tiling"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_metadata_json": ("STRING", {"multiline": True}),
                "blend_masks": ("MASK",),
                "scale_mode": (["infer_from_tile_size", "metadata_scale_factor"], {"default": "infer_from_tile_size"}),
            },
        }

    def blend(self, tiles, tile_metadata_json, blend_masks, scale_mode):
        try:
            metadata = json.loads(tile_metadata_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Tile metadata must be valid JSON from Smart Tile Planner.") from exc

        tile_records = metadata.get("tiles", [])
        if not tile_records:
            raise ValueError("Tile metadata does not contain any tiles.")
        if len(tile_records) != tiles.shape[0]:
            raise ValueError(
                f"Metadata describes {len(tile_records)} tiles, but the processed batch contains {tiles.shape[0]}."
            )
        if blend_masks.shape[0] != tiles.shape[0]:
            raise ValueError(
                f"Received {blend_masks.shape[0]} masks for {tiles.shape[0]} processed tiles."
            )

        planned_tile_width = metadata["tile_width"]
        planned_tile_height = metadata["tile_height"]
        output_tile_height = tiles.shape[1]
        output_tile_width = tiles.shape[2]

        explicit_output_size = (
            metadata.get("output_tile_width"),
            metadata.get("output_tile_height"),
        )
        matches_explicit_size = explicit_output_size == (output_tile_width, output_tile_height)

        if scale_mode == "metadata_scale_factor" or matches_explicit_size:
            scale_x = scale_y = float(metadata.get("scale_factor", 1.0))
            expected_width = int(
                metadata.get("output_tile_width", round(planned_tile_width * scale_x))
            )
            expected_height = int(
                metadata.get("output_tile_height", round(planned_tile_height * scale_y))
            )
            if (output_tile_width, output_tile_height) != (expected_width, expected_height):
                raise ValueError(
                    "Processed tile dimensions do not match the planner scale factor: "
                    f"expected {expected_width}x{expected_height}, received "
                    f"{output_tile_width}x{output_tile_height}."
                )
        else:
            scale_x = output_tile_width / planned_tile_width
            scale_y = output_tile_height / planned_tile_height

        output_width = int(metadata.get("output_width", max(1, round(metadata["image_width"] * scale_x))))
        output_height = int(metadata.get("output_height", max(1, round(metadata["image_height"] * scale_y))))
        channels = tiles.shape[3]
        device = tiles.device
        dtype = tiles.dtype

        image_batch = int(metadata.get("image_batch", 1))
        _stitch_memory_guard(output_width, output_height, channels, image_batch)
        canvas = torch.zeros(
            (image_batch, output_height, output_width, channels),
            dtype=dtype,
            device=device,
        )
        weights = torch.zeros(
            (image_batch, output_height, output_width, 1),
            dtype=dtype,
            device=device,
        )

        for tile_record in tile_records:
            tile_index = tile_record["tile_index"]
            source_index = tile_record.get("source_index", 0)
            if not 0 <= source_index < image_batch:
                raise ValueError(f"Tile {tile_index} has invalid source index {source_index}.")
            tile = tiles[tile_index]
            mask = blend_masks[tile_index].to(device=device, dtype=dtype)

            # End-anchored rounding: the tile's end comes from its absolute
            # source end, so fractional scales can never open 1px gaps between
            # neighboring tiles.
            x0 = round(tile_record["x"] * scale_x)
            y0 = round(tile_record["y"] * scale_y)
            valid_width = max(
                1, round((tile_record["x"] + tile_record["width"]) * scale_x) - x0
            )
            valid_height = max(
                1, round((tile_record["y"] + tile_record["height"]) * scale_y) - y0
            )
            x1 = min(output_width, x0 + valid_width)
            y1 = min(output_height, y0 + valid_height)
            if x1 <= x0 or y1 <= y0:
                continue

            tile_crop = tile[: y1 - y0, : x1 - x0, :]
            resized_mask = F.interpolate(
                mask[None, None, :, :],
                size=(output_tile_height, output_tile_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            mask_crop = resized_mask[: y1 - y0, : x1 - x0].unsqueeze(-1).clamp(0.0, 1.0)

            canvas[source_index, y0:y1, x0:x1, :] += tile_crop * mask_crop
            weights[source_index, y0:y1, x0:x1, :] += mask_crop

        # min() is a scalar reduction and the remaining math runs in place, so
        # peak memory stays near one canvas + one weight plane even at extreme
        # output sizes (a gigapixel canvas must not be copied for the division).
        if float(weights.min()) <= 1e-6:
            raise ValueError("Blend masks leave uncovered pixels in the stitched image.")
        canvas.div_(weights)
        return (canvas.clamp_(0.0, 1.0),)
