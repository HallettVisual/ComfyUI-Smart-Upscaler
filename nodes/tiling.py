import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def _positions(length, tile_size, overlap):
    if length <= tile_size:
        return [0]

    step = max(1, tile_size - overlap)
    values = list(range(0, max(1, length - tile_size + 1), step))
    last = length - tile_size
    if values[-1] != last:
        values.append(last)
    return values


def _pad_image(tile, target_height, target_width, mode):
    height, width, channels = tile.shape
    pad_bottom = target_height - height
    pad_right = target_width - width
    if pad_bottom <= 0 and pad_right <= 0:
        return tile

    tensor = tile.permute(2, 0, 1).unsqueeze(0)
    padding = (0, pad_right, 0, pad_bottom)
    if mode == "zero":
        padded = F.pad(tensor, padding, mode="constant", value=0.0)
    else:
        padded = F.pad(tensor, padding, mode="replicate")
    return padded.squeeze(0).permute(1, 2, 0).contiguous()


def _edge_ramp(length, feather, fade_start, fade_end, device):
    ramp = torch.ones(length, dtype=torch.float32, device=device)
    feather = min(max(0, feather), length)
    if feather == 0:
        return ramp

    if fade_start:
        ramp[:feather] = torch.linspace(0.0, 1.0, feather, device=device)
    if fade_end:
        ramp[-feather:] = torch.linspace(1.0, 0.0, feather, device=device)
    return ramp


def _blend_mask(
    tile_height,
    tile_width,
    valid_height,
    valid_width,
    x0,
    y0,
    image_width,
    image_height,
    feather,
    overlap,
    device,
):
    effective_feather = min(feather, overlap)
    horizontal = _edge_ramp(
        valid_width,
        effective_feather,
        fade_start=x0 > 0,
        fade_end=(x0 + valid_width) < image_width,
        device=device,
    )
    vertical = _edge_ramp(
        valid_height,
        effective_feather,
        fade_start=y0 > 0,
        fade_end=(y0 + valid_height) < image_height,
        device=device,
    )
    valid_mask = vertical[:, None] * horizontal[None, :]
    mask = torch.zeros((tile_height, tile_width), dtype=torch.float32, device=device)
    mask[:valid_height, :valid_width] = valid_mask
    return mask


def _draw_grid(preview, tiles):
    output = preview.clone()
    colors = torch.tensor(
        [
            [1.0, 0.16, 0.16],
            [0.1, 0.75, 1.0],
            [1.0, 0.8, 0.12],
            [0.25, 1.0, 0.45],
        ],
        dtype=output.dtype,
        device=output.device,
    )
    height, width, _ = output.shape
    line_width = max(1, math.ceil(max(height, width) / 700))

    for tile in tiles:
        color = colors[tile["tile_index"] % len(colors)]
        x0 = tile.get("core_x", tile["x"])
        y0 = tile.get("core_y", tile["y"])
        x1 = x0 + tile.get("core_width", tile["width"])
        y1 = y0 + tile.get("core_height", tile["height"])
        output[y0 : min(y0 + line_width, height), x0:x1] = color
        output[max(y1 - line_width, 0) : y1, x0:x1] = color
        output[y0:y1, x0 : min(x0 + line_width, width)] = color
        output[y0:y1, max(x1 - line_width, 0) : x1] = color

    return output


def _draw_adaptive_processing_preview(
    source,
    tiles,
    scale_factor,
    output_overlap,
    output_width,
    output_height,
    output_feather=0,
    max_preview_size=1600,
):
    """Draw the actual expanded processing crops on a lightweight output-scale preview."""
    preview_scale = min(1.0, max_preview_size / max(output_width, output_height))
    preview_width = max(1, round(output_width * preview_scale))
    preview_height = max(1, round(output_height * preview_scale))
    output = F.interpolate(
        source.permute(2, 0, 1).unsqueeze(0),
        size=(preview_height, preview_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0).permute(1, 2, 0).contiguous()

    colors = torch.tensor(
        [
            [1.0, 0.16, 0.16],
            [0.1, 0.75, 1.0],
            [1.0, 0.8, 0.12],
            [0.25, 1.0, 0.45],
        ],
        dtype=output.dtype,
        device=output.device,
    )
    vertical_overlap_color = torch.tensor(
        [1.0, 0.1, 0.75], dtype=output.dtype, device=output.device
    )
    horizontal_overlap_color = torch.tensor(
        [0.1, 0.9, 1.0], dtype=output.dtype, device=output.device
    )
    seam_color = torch.ones(3, dtype=output.dtype, device=output.device)
    feather_color = torch.tensor(
        [1.0, 0.82, 0.08], dtype=output.dtype, device=output.device
    )
    line_width = max(1, math.ceil(max(preview_height, preview_width) / 800))
    coordinate_scale = scale_factor * preview_scale
    overlap_before = round((output_overlap // 2) * preview_scale)
    overlap_after = round((output_overlap - output_overlap // 2) * preview_scale)
    feather_width = max(0, round(output_feather * preview_scale))

    columns = max((tile["column"] for tile in tiles), default=0) + 1
    rows = max((tile["row"] for tile in tiles), default=0) + 1
    vertical_seams = sorted(
        {
            tile["core_x"] + tile["core_width"]
            for tile in tiles
            if tile["column"] < columns - 1
        }
    )
    horizontal_seams = sorted(
        {
            tile["core_y"] + tile["core_height"]
            for tile in tiles
            if tile["row"] < rows - 1
        }
    )

    def tint_region(x0, y0, x1, y1, color, alpha=0.34):
        x0 = max(0, min(preview_width, x0))
        x1 = max(0, min(preview_width, x1))
        y0 = max(0, min(preview_height, y0))
        y1 = max(0, min(preview_height, y1))
        if x1 > x0 and y1 > y0:
            output[y0:y1, x0:x1] = output[y0:y1, x0:x1] * (1.0 - alpha) + color * alpha

    def tint_vertical_gradient(x0, x1, reverse=False):
        x0 = max(0, min(preview_width, x0))
        x1 = max(0, min(preview_width, x1))
        if x1 <= x0:
            return
        alpha = torch.linspace(0.46, 0.04, x1 - x0, dtype=output.dtype, device=output.device)
        if reverse:
            alpha = alpha.flip(0)
        alpha = alpha.view(1, -1, 1)
        output[:, x0:x1] = output[:, x0:x1] * (1.0 - alpha) + feather_color * alpha

    def tint_horizontal_gradient(y0, y1, reverse=False):
        y0 = max(0, min(preview_height, y0))
        y1 = max(0, min(preview_height, y1))
        if y1 <= y0:
            return
        alpha = torch.linspace(0.46, 0.04, y1 - y0, dtype=output.dtype, device=output.device)
        if reverse:
            alpha = alpha.flip(0)
        alpha = alpha.view(-1, 1, 1)
        output[y0:y1] = output[y0:y1] * (1.0 - alpha) + feather_color * alpha

    for seam in vertical_seams:
        x = round(seam * coordinate_scale)
        tint_region(
            x - overlap_before,
            0,
            x + overlap_after,
            preview_height,
            vertical_overlap_color,
        )
    for seam in horizontal_seams:
        y = round(seam * coordinate_scale)
        tint_region(
            0,
            y - overlap_before,
            preview_width,
            y + overlap_after,
            horizontal_overlap_color,
        )

    # Yellow gradients show the exact edge-mask ramps used by the blender.
    if feather_width:
        for seam in vertical_seams:
            x = round(seam * coordinate_scale)
            band_start = x - overlap_before
            band_end = x + overlap_after
            tint_vertical_gradient(band_start, band_start + feather_width)
            tint_vertical_gradient(band_end - feather_width, band_end, reverse=True)
        for seam in horizontal_seams:
            y = round(seam * coordinate_scale)
            band_start = y - overlap_before
            band_end = y + overlap_after
            tint_horizontal_gradient(band_start, band_start + feather_width)
            tint_horizontal_gradient(band_end - feather_width, band_end, reverse=True)

    # Colored rectangles are the expanded crops actually sent to the upscaler and sampler.
    for tile in tiles:
        color = colors[tile["tile_index"] % len(colors)]
        x0 = max(0, min(preview_width, round(tile["x"] * coordinate_scale)))
        y0 = max(0, min(preview_height, round(tile["y"] * coordinate_scale)))
        x1 = max(0, min(preview_width, round((tile["x"] + tile["width"]) * coordinate_scale)))
        y1 = max(0, min(preview_height, round((tile["y"] + tile["height"]) * coordinate_scale)))
        output[y0 : min(y0 + line_width, preview_height), x0:x1] = color
        output[max(y1 - line_width, 0) : y1, x0:x1] = color
        output[y0:y1, x0 : min(x0 + line_width, preview_width)] = color
        output[y0:y1, max(x1 - line_width, 0) : x1] = color

    # White center lines show where neighboring core regions meet inside the shaded overlap.
    for seam in vertical_seams:
        x = max(0, min(preview_width - 1, round(seam * coordinate_scale)))
        output[:, x : min(x + line_width, preview_width)] = seam_color
    for seam in horizontal_seams:
        y = max(0, min(preview_height - 1, round(seam * coordinate_scale)))
        output[y : min(y + line_width, preview_height), :] = seam_color

    # Tile identity is always T001, T002, ... across the overlay, prompts, and reviewer.
    output = output.clamp(0.0, 1.0)
    output_array = (output.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    output_image = Image.fromarray(output_array, mode="RGB")
    draw = ImageDraw.Draw(output_image)
    # Text scales with the preview so labels stay readable at any size (the
    # PIL default bitmap font is ~10px regardless of image size).
    font_size = max(16, round(max(preview_width, preview_height) / 45))
    font = None
    for candidate in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except OSError:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()
    pad = max(3, font_size // 5)
    legend = (
        f"OUTPUT {int(output_width)}x{int(output_height)}  |  "
        f"OVERLAP {int(output_overlap)} PX  |  FEATHER {int(output_feather)} PX"
    )
    legend_box = draw.textbbox((8, 8), legend, font=font)
    draw.rectangle(
        (
            legend_box[0] - pad,
            legend_box[1] - pad,
            legend_box[2] + pad,
            legend_box[3] + pad,
        ),
        fill=(15, 17, 20),
    )
    draw.text((8, 8), legend, fill=(255, 255, 255), font=font)
    label_offset = legend_box[3] + pad * 2 + 4
    for tile in tiles:
        x = max(3, round(tile["core_x"] * coordinate_scale) + 6)
        y = max(label_offset if tile["core_y"] == 0 else 3,
                round(tile["core_y"] * coordinate_scale) + 6)
        label = f"T{int(tile['tile_index']) + 1:03d}"
        text_box = draw.textbbox((x, y), label, font=font)
        draw.rectangle(
            (
                text_box[0] - pad,
                text_box[1] - pad,
                text_box[2] + pad,
                text_box[3] + pad,
            ),
            fill=(15, 17, 20),
        )
        draw.text((x, y), label, fill=(255, 255, 255), font=font)
    labeled = torch.from_numpy(np.asarray(output_image).copy()).to(
        device=output.device,
        dtype=output.dtype,
    )
    return labeled / 255.0


def _round_up(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def _adaptive_axis(length, scale_factor, min_tile_size, max_tile_size, overlap, divisible_by):
    scaled_length = max(1, round(length * scale_factor))
    effective_min = _round_up(min_tile_size, divisible_by)
    effective_max = max(divisible_by, (max_tile_size // divisible_by) * divisible_by)
    if effective_min > effective_max:
        raise ValueError("Minimum tile size must not exceed maximum tile size after divisibility rounding.")

    half_before_output = overlap // 2
    half_after_output = overlap - half_before_output
    before_source = math.ceil(half_before_output / scale_factor)
    after_source = math.ceil(half_after_output / scale_factor)

    def layout(count):
        boundaries = [round(index * length / count) for index in range(count + 1)]
        max_crop = 0
        for index in range(count):
            core_start = boundaries[index]
            core_end = boundaries[index + 1]
            crop_start = max(0, core_start - (before_source if index > 0 else 0))
            crop_end = min(length, core_end + (after_source if index < count - 1 else 0))
            max_crop = max(max_crop, crop_end - crop_start)
        output_tile_size = _round_up(
            max(effective_min, math.ceil(max_crop * scale_factor)),
            divisible_by,
        )
        source_tile_size = max(max_crop, math.ceil(output_tile_size / scale_factor))
        return boundaries, source_tile_size, output_tile_size

    usable_core = max(divisible_by, effective_max - overlap)
    count = max(1, math.ceil(scaled_length / usable_core))
    boundaries, source_tile_size, output_tile_size = layout(count)
    while output_tile_size > effective_max and count < length:
        count += 1
        boundaries, source_tile_size, output_tile_size = layout(count)

    if output_tile_size > effective_max:
        raise ValueError("Could not create divisible tiles within the requested maximum size.")

    return {
        "count": count,
        "boundaries": boundaries,
        "before_source": before_source,
        "after_source": after_source,
        "source_tile_size": source_tile_size,
        "output_tile_size": output_tile_size,
        "effective_min": effective_min,
        "effective_max": effective_max,
    }


class SmartTilePlanner:
    CATEGORY = "Smart Upscaler/Tiling"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("tiles", "grid_preview", "blend_masks", "tile_metadata_json")
    FUNCTION = "plan"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 64}),
                "tile_height": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 96, "min": 0, "max": 2048, "step": 8}),
                "feather": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8}),
                "scale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 12.0, "step": 0.25}),
                "padding_mode": (["edge", "zero"], {"default": "edge"}),
            },
        }

    def plan(self, image, tile_width, tile_height, overlap, feather, scale_factor, padding_mode):
        if overlap >= tile_width or overlap >= tile_height:
            raise ValueError("Overlap must be smaller than both tile dimensions.")

        batch, image_height, image_width, channels = image.shape
        tiles = []
        masks = []
        metadata_tiles = []

        tile_index = 0
        for source_index in range(batch):
            source = image[source_index]
            x_values = _positions(image_width, tile_width, overlap)
            y_values = _positions(image_height, tile_height, overlap)

            for row, y0 in enumerate(y_values):
                for col, x0 in enumerate(x_values):
                    x1 = min(x0 + tile_width, image_width)
                    y1 = min(y0 + tile_height, image_height)
                    crop = source[y0:y1, x0:x1, :]
                    valid_height = y1 - y0
                    valid_width = x1 - x0
                    tiles.append(_pad_image(crop, tile_height, tile_width, padding_mode))
                    masks.append(
                        _blend_mask(
                            tile_height,
                            tile_width,
                            valid_height,
                            valid_width,
                            x0,
                            y0,
                            image_width,
                            image_height,
                            feather,
                            overlap,
                            image.device,
                        )
                    )
                    metadata_tiles.append(
                        {
                            "tile_index": tile_index,
                            "source_index": source_index,
                            "row": row,
                            "column": col,
                            "x": x0,
                            "y": y0,
                            "width": valid_width,
                            "height": valid_height,
                            "padded_width": tile_width,
                            "padded_height": tile_height,
                            "position": self._position_label(
                                x0,
                                y0,
                                valid_width,
                                valid_height,
                                image_width,
                                image_height,
                            ),
                        }
                    )
                    tile_index += 1

        if not tiles:
            raise ValueError("No tiles were produced.")

        preview = _draw_grid(image[0], [tile for tile in metadata_tiles if tile["source_index"] == 0])
        metadata = {
            "version": 1,
            "image_width": image_width,
            "image_height": image_height,
            "image_batch": batch,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "overlap": overlap,
            "feather": feather,
            "scale_factor": scale_factor,
            "tile_count": len(metadata_tiles),
            "tiles": metadata_tiles,
        }
        return (
            torch.stack(tiles, dim=0),
            preview.unsqueeze(0),
            torch.stack(masks, dim=0),
            json.dumps(metadata, indent=2),
        )

    @staticmethod
    def _position_label(x, y, tile_width, tile_height, width, height):
        x_center = ((x + tile_width / 2) / max(1, width))
        y_center = ((y + tile_height / 2) / max(1, height))
        horizontal = "left" if x_center < 0.34 else "right" if x_center > 0.66 else "center"
        vertical = "top" if y_center < 0.34 else "bottom" if y_center > 0.66 else "middle"
        if horizontal == "center" and vertical == "middle":
            return "center"
        return f"{vertical} {horizontal}"


class SmartAdaptiveTilePlanner:
    CATEGORY = "Smart Upscaler/Tiling"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("tiles", "grid_preview", "blend_masks", "tile_metadata_json")
    FUNCTION = "plan"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_tile_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Minimum processed tile size in sampler output pixels.",
                    },
                ),
                "max_tile_size": (
                    "INT",
                    {
                        "default": 1536,
                        "min": 256,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Maximum processed tile size in sampler output pixels.",
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 1024,
                        "step": 8,
                        "tooltip": "Total shared overlap across an internal tile edge, in output pixels.",
                    },
                ),
                "feather": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 512,
                        "step": 8,
                        "tooltip": "Blend fade width at internal tile edges, in output pixels.",
                    },
                ),
                "scale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 12.0, "step": 0.25}),
                "divisible_by": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
                "padding_mode": (["edge", "zero"], {"default": "edge"}),
            },
        }

    def plan(
        self,
        image,
        min_tile_size,
        max_tile_size,
        overlap,
        feather,
        scale_factor,
        divisible_by,
        padding_mode,
    ):
        if min_tile_size > max_tile_size:
            raise ValueError("Minimum tile size must not exceed maximum tile size.")
        if feather > overlap:
            raise ValueError("Feather must be less than or equal to overlap.")
        if overlap >= max_tile_size:
            raise ValueError("Overlap must be smaller than maximum tile size.")

        batch, image_height, image_width, _ = image.shape
        x_axis = _adaptive_axis(
            image_width,
            scale_factor,
            min_tile_size,
            max_tile_size,
            overlap,
            divisible_by,
        )
        y_axis = _adaptive_axis(
            image_height,
            scale_factor,
            min_tile_size,
            max_tile_size,
            overlap,
            divisible_by,
        )

        overlap_source = max(
            x_axis["before_source"] + x_axis["after_source"],
            y_axis["before_source"] + y_axis["after_source"],
        )
        feather_source = math.ceil(feather / scale_factor)
        tiles = []
        masks = []
        metadata_tiles = []
        tile_index = 0

        for source_index in range(batch):
            source = image[source_index]
            for row in range(y_axis["count"]):
                core_y0 = y_axis["boundaries"][row]
                core_y1 = y_axis["boundaries"][row + 1]
                y0 = max(0, core_y0 - (y_axis["before_source"] if row > 0 else 0))
                y1 = min(
                    image_height,
                    core_y1 + (y_axis["after_source"] if row < y_axis["count"] - 1 else 0),
                )

                for column in range(x_axis["count"]):
                    core_x0 = x_axis["boundaries"][column]
                    core_x1 = x_axis["boundaries"][column + 1]
                    x0 = max(0, core_x0 - (x_axis["before_source"] if column > 0 else 0))
                    x1 = min(
                        image_width,
                        core_x1 + (x_axis["after_source"] if column < x_axis["count"] - 1 else 0),
                    )

                    valid_width = x1 - x0
                    valid_height = y1 - y0
                    crop = source[y0:y1, x0:x1, :]
                    tiles.append(
                        _pad_image(
                            crop,
                            y_axis["source_tile_size"],
                            x_axis["source_tile_size"],
                            padding_mode,
                        )
                    )
                    masks.append(
                        _blend_mask(
                            y_axis["source_tile_size"],
                            x_axis["source_tile_size"],
                            valid_height,
                            valid_width,
                            x0,
                            y0,
                            image_width,
                            image_height,
                            feather_source,
                            overlap_source,
                            image.device,
                        )
                    )
                    metadata_tiles.append(
                        {
                            "tile_index": tile_index,
                            "source_index": source_index,
                            "row": row,
                            "column": column,
                            "x": x0,
                            "y": y0,
                            "width": valid_width,
                            "height": valid_height,
                            "core_x": core_x0,
                            "core_y": core_y0,
                            "core_width": core_x1 - core_x0,
                            "core_height": core_y1 - core_y0,
                            "padded_width": x_axis["source_tile_size"],
                            "padded_height": y_axis["source_tile_size"],
                            "output_width": x_axis["output_tile_size"],
                            "output_height": y_axis["output_tile_size"],
                            "position": SmartTilePlanner._position_label(
                                core_x0,
                                core_y0,
                                core_x1 - core_x0,
                                core_y1 - core_y0,
                                image_width,
                                image_height,
                            ),
                        }
                    )
                    tile_index += 1

        output_width = round(image_width * scale_factor)
        output_height = round(image_height * scale_factor)
        preview = _draw_adaptive_processing_preview(
            image[0],
            [tile for tile in metadata_tiles if tile["source_index"] == 0],
            scale_factor,
            overlap,
            output_width,
            output_height,
            output_feather=feather,
        )
        metadata = {
            "version": 2,
            "planner": "adaptive_even_grid",
            "tile_size_units": "output_pixels",
            "image_width": image_width,
            "image_height": image_height,
            "image_batch": batch,
            "output_width": output_width,
            "output_height": output_height,
            "preview": "low_resolution_output_canvas_with_overlap_bands_and_processing_crop_outlines",
            "tile_width": x_axis["source_tile_size"],
            "tile_height": y_axis["source_tile_size"],
            "output_tile_width": x_axis["output_tile_size"],
            "output_tile_height": y_axis["output_tile_size"],
            "min_tile_size": x_axis["effective_min"],
            "max_tile_size": x_axis["effective_max"],
            "overlap": overlap_source,
            "feather": feather_source,
            "output_overlap": overlap,
            "output_feather": feather,
            "scale_factor": scale_factor,
            "divisible_by": divisible_by,
            "grid_columns": x_axis["count"],
            "grid_rows": y_axis["count"],
            "tile_count": len(metadata_tiles),
            "tiles": metadata_tiles,
        }
        return (
            torch.stack(tiles, dim=0),
            preview.unsqueeze(0),
            torch.stack(masks, dim=0),
            json.dumps(metadata, indent=2),
        )
