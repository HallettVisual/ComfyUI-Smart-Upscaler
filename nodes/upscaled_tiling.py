import json

import torch
import torch.nn.functional as F

from .blending import SmartTileBlender
from .cache import esrgan_cache_key, load_esrgan_tiles, save_esrgan_tiles
from .tiling import SmartAdaptiveTilePlanner, _draw_adaptive_processing_preview


UPSCALE_METHODS = (
    "AI upscaler model (ESRGAN, etc.)",
    "Lanczos (sharp standard resize)",
    "Bicubic (balanced standard resize)",
    "Bilinear (gentle standard resize)",
    "Area (soft standard resize)",
)

_STANDARD_RESIZE_METHODS = {
    UPSCALE_METHODS[1]: "lanczos",
    UPSCALE_METHODS[2]: "bicubic",
    UPSCALE_METHODS[3]: "bilinear",
    UPSCALE_METHODS[4]: "area",
}


def _upscale_tiles_with_model(
    upscale_model,
    source_tiles,
    target_width,
    target_height,
    upscale_batch_size,
):
    from comfy import model_management
    import comfy.utils

    device = model_management.get_torch_device()
    chunk_size = max(1, int(upscale_batch_size))
    example_count = min(chunk_size, source_tiles.shape[0])
    memory_required = model_management.module_size(upscale_model.model)
    memory_required += (
        512
        * 512
        * 3
        * source_tiles.element_size()
        * max(float(upscale_model.scale), 1.0)
        * 384.0
        * example_count
    )
    model_management.free_memory(memory_required, device)

    resized_chunks = []
    upscale_model.to(device)
    output_device = model_management.intermediate_device()
    try:
        for start in range(0, source_tiles.shape[0], chunk_size):
            chunk = source_tiles[start : start + chunk_size].movedim(-1, -3).to(device)
            tile_size = 512
            while True:
                try:
                    steps = chunk.shape[0] * comfy.utils.get_tiled_scale_steps(
                        chunk.shape[3],
                        chunk.shape[2],
                        tile_x=tile_size,
                        tile_y=tile_size,
                        overlap=32,
                    )
                    progress = comfy.utils.ProgressBar(steps)
                    upscaled = comfy.utils.tiled_scale(
                        chunk,
                        lambda value: upscale_model(value.float()),
                        tile_x=tile_size,
                        tile_y=tile_size,
                        overlap=32,
                        upscale_amount=upscale_model.scale,
                        pbar=progress,
                        output_device=output_device,
                    )
                    break
                except Exception as exc:
                    model_management.raise_non_oom(exc)
                    tile_size //= 2
                    if tile_size < 128:
                        raise

            upscaled = upscaled.float()
            if upscaled.shape[2:] != (target_height, target_width):
                upscaled = F.interpolate(
                    upscaled,
                    size=(target_height, target_width),
                    mode="area",
                )
            resized_chunks.append(
                upscaled.movedim(-3, -1).clamp(0.0, 1.0).to(
                    device=output_device,
                    dtype=model_management.intermediate_dtype(),
                )
            )
    finally:
        upscale_model.to("cpu")

    return torch.cat(resized_chunks, dim=0)


def _resize_tiles_without_model(source_tiles, target_width, target_height, method):
    """Use ComfyUI's standard image resizing without an AI upscaler model."""

    import comfy.utils

    resized = comfy.utils.common_upscale(
        source_tiles.movedim(-1, -3),
        int(target_width),
        int(target_height),
        str(method),
        "disabled",
    )
    return resized.movedim(-3, -1).clamp(0.0, 1.0)


def _output_scale_blend_masks(metadata, device, dtype):
    """Build blend masks directly in output pixels from the planned geometry.

    Masks resized up from source scale drift whenever the output tile size is
    not an exact multiple of the padded source tile (any fractional effective
    scale, e.g. 6x with divisible-by rounding), which pushed mask zeros into
    covered pixels and broke stitching. Building in output space is exact at
    every scale: inside each tile's valid area the mask is 1, with a linear
    feather ramp only on edges that actually have a neighbor, so coverage can
    never have holes.
    """
    records = sorted(metadata["tiles"], key=lambda item: int(item["tile_index"]))
    tile_height = int(metadata["output_tile_height"])
    tile_width = int(metadata["output_tile_width"])
    scale = float(metadata.get("scale_factor", 1.0))
    overlap = int(metadata.get("output_overlap", 0))
    feather = int(metadata.get("output_feather", 0))
    # Both neighbors ramp to zero at opposite ends of the shared overlap; keeping
    # each ramp within half the overlap guarantees their weights never both
    # vanish on the same pixel.
    ramp = min(feather, max(1, overlap // 2)) if overlap > 0 and feather > 0 else 0

    by_source = {}
    for record in records:
        by_source.setdefault(int(record.get("source_index", 0)), set()).add(
            (int(record["row"]), int(record["column"]))
        )

    masks = torch.zeros((len(records), tile_height, tile_width), dtype=dtype, device=device)
    for record in records:
        grid = by_source[int(record.get("source_index", 0))]
        row, column = int(record["row"]), int(record["column"])
        x0 = round(float(record["x"]) * scale)
        y0 = round(float(record["y"]) * scale)
        valid_width = min(
            tile_width,
            max(1, round((float(record["x"]) + float(record["width"])) * scale) - x0),
        )
        valid_height = min(
            tile_height,
            max(1, round((float(record["y"]) + float(record["height"])) * scale) - y0),
        )

        profile_x = torch.ones(valid_width, dtype=dtype, device=device)
        profile_y = torch.ones(valid_height, dtype=dtype, device=device)
        if ramp > 0:
            rise = torch.linspace(0.0, 1.0, ramp, dtype=dtype, device=device)
            if (row, column - 1) in grid and ramp <= valid_width:
                profile_x[:ramp] = torch.minimum(profile_x[:ramp], rise)
            if (row, column + 1) in grid and ramp <= valid_width:
                profile_x[valid_width - ramp :] = torch.minimum(
                    profile_x[valid_width - ramp :], rise.flip(0)
                )
            if (row - 1, column) in grid and ramp <= valid_height:
                profile_y[:ramp] = torch.minimum(profile_y[:ramp], rise)
            if (row + 1, column) in grid and ramp <= valid_height:
                profile_y[valid_height - ramp :] = torch.minimum(
                    profile_y[valid_height - ramp :], rise.flip(0)
                )
        masks[int(record["tile_index"]), :valid_height, :valid_width] = (
            profile_y.view(-1, 1) * profile_x.view(1, -1)
        )
    return masks


class SmartUpscaledTilePlanner:
    CATEGORY = "Smart Upscaler/Tiling"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = (
        "upscaled_tiles",
        "processing_preview",
        "upscaled_image",
        "blend_masks",
        "tile_metadata_json",
        "preflight_summary",
    )
    FUNCTION = "plan_and_upscale"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (
                    list(UPSCALE_METHODS),
                    {
                        "default": UPSCALE_METHODS[0],
                        "label": "How Should the Source Be Enlarged?",
                        "tooltip": "AI models can add detail but may round small objects. Standard resize methods preserve source geometry and need no model.",
                    },
                ),
                "min_tile_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Minimum processing tile size in final sampler pixels.",
                    },
                ),
                "max_tile_size": (
                    "INT",
                    {
                        "default": 1536,
                        "min": 256,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Maximum processing tile size in final sampler pixels.",
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 1024,
                        "step": 8,
                        "tooltip": "Total shared overlap in final sampler pixels.",
                    },
                ),
                "feather": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 512,
                        "step": 8,
                        "tooltip": "Blend fade width in final sampler pixels.",
                    },
                ),
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 12.0,
                        "step": 0.25,
                        "tooltip": "How many times larger the final image is. High scales multiply tile count and time: a 1340x896 source at 12x renders about 90 tiles.",
                    },
                ),
                "divisible_by": (
                    "INT",
                    {"default": 16, "min": 1, "max": 256, "step": 1},
                ),
                "upscale_batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "advanced": True,
                        "label": "AI Upscaler Batch Size (Advanced)",
                        "tooltip": "Tiles sent through an AI upscaler together. Ignored by standard resize methods.",
                    },
                ),
                "padding_mode": (
                    ["edge", "zero"],
                    {
                        "default": "edge",
                        "advanced": True,
                        "label": "Border Padding (Advanced)",
                        "tooltip": "How tiles that reach past the image border are filled before enlargement. Edge repeats the outermost pixels (right for photos - leave it here). Zero fills black, only useful for images with true black borders.",
                    },
                ),
                "cache_mode": (
                    ["read_write", "refresh", "bypass"],
                    {
                        "default": "read_write",
                        "advanced": True,
                        "tooltip": "Reuse enlarged tiles from disk, regenerate them, or skip the disk cache.",
                    },
                ),
                "cache_tag": (
                    "STRING",
                    {
                        "default": "auto",
                        "advanced": True,
                        "tooltip": "Optional cache label. The upscaler weights are fingerprinted automatically.",
                    },
                ),
                "preview_max_size": (
                    "INT",
                    {
                        "default": 1600,
                        "min": 512,
                        "max": 8192,
                        "step": 128,
                        "advanced": True,
                        "tooltip": "Maximum dimension of the output-scale diagnostic preview.",
                    },
                ),
            },
            "optional": {
                "upscale_model": (
                    "UPSCALE_MODEL",
                    {
                        "lazy": True,
                        "tooltip": "Required only when the AI upscaler method is selected.",
                    },
                ),
            },
        }

    def check_lazy_status(
        self,
        image,
        upscale_method,
        min_tile_size,
        max_tile_size,
        overlap,
        feather,
        scale_factor,
        divisible_by,
        upscale_batch_size,
        padding_mode,
        cache_mode="read_write",
        cache_tag="auto",
        preview_max_size=1600,
        upscale_model=None,
    ):
        if str(upscale_method) == UPSCALE_METHODS[0] and upscale_model is None:
            return ["upscale_model"]
        return []

    def plan_and_upscale(
        self,
        image,
        min_tile_size,
        max_tile_size,
        overlap,
        feather,
        scale_factor,
        divisible_by,
        upscale_batch_size,
        padding_mode,
        cache_mode="read_write",
        cache_tag="auto",
        preview_max_size=1600,
        upscale_method=UPSCALE_METHODS[0],
        upscale_model=None,
    ):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("Smart Upscaler expected one ComfyUI IMAGE tensor.")
        if int(image.shape[0]) != 1:
            raise ValueError(
                "Smart Upscaler processes one source image at a time. Split image "
                "batches before this node so scene prompts, caches, and tile "
                "continuity can never cross between unrelated images."
            )
        source_tiles, _, masks, metadata_json = SmartAdaptiveTilePlanner().plan(
            image,
            min_tile_size,
            max_tile_size,
            overlap,
            feather,
            scale_factor,
            divisible_by,
            padding_mode,
        )
        metadata = json.loads(metadata_json)
        target_width = int(metadata["output_tile_width"])
        target_height = int(metadata["output_tile_height"])
        expected_shape = (
            int(source_tiles.shape[0]),
            target_height,
            target_width,
            int(source_tiles.shape[3]),
        )
        settings = {
            "min_tile_size": int(min_tile_size),
            "max_tile_size": int(max_tile_size),
            "overlap": int(overlap),
            "feather": int(feather),
            "scale_factor": float(scale_factor),
            "divisible_by": int(divisible_by),
            "padding_mode": str(padding_mode),
            "target_width": target_width,
            "target_height": target_height,
            "upscale_method": str(upscale_method),
        }
        cache_key = None
        cached_tiles = None
        cache_model = upscale_model if str(upscale_method) == UPSCALE_METHODS[0] else None
        if cache_mode != "bypass":
            cache_key = esrgan_cache_key(image, cache_model, settings, cache_tag)
            if cache_mode == "read_write":
                cached_tiles = load_esrgan_tiles(cache_key, expected_shape)

        if cached_tiles is not None:
            upscaled_tiles = cached_tiles.to(
                device=source_tiles.device,
                dtype=source_tiles.dtype,
            )
            cache_status = "HIT"
        else:
            if str(upscale_method) == UPSCALE_METHODS[0]:
                if upscale_model is None:
                    raise ValueError(
                        "Connect an UPSCALE_MODEL or choose Lanczos, Bicubic, Bilinear, or Area."
                    )
                upscaled_tiles = _upscale_tiles_with_model(
                    upscale_model,
                    source_tiles,
                    target_width,
                    target_height,
                    upscale_batch_size,
                )
            else:
                resize_method = _STANDARD_RESIZE_METHODS.get(str(upscale_method))
                if resize_method is None:
                    raise ValueError(f"Unknown upscale method: {upscale_method}")
                upscaled_tiles = _resize_tiles_without_model(
                    source_tiles,
                    target_width,
                    target_height,
                    resize_method,
                )
            if cache_mode != "bypass":
                saved_path, cache_error = save_esrgan_tiles(cache_key, upscaled_tiles)
                cache_status = (
                    "WRITE"
                    if saved_path is not None
                    else f"WRITE SKIPPED ({cache_error})"
                )
            else:
                cache_status = "BYPASS"

        metadata["version"] = 3
        metadata["planner"] = "adaptive_output_scale_grid"
        metadata["tiles_are_output_scale"] = True
        metadata["upscale_method"] = str(upscale_method)
        metadata["upscale_model_scale"] = float(getattr(cache_model, "scale", 1.0))
        metadata["upscale_batch_size"] = int(upscale_batch_size)
        metadata["preprocess_cache"] = {
            "mode": str(cache_mode),
            "status": cache_status,
            "key": cache_key[:12] if cache_key else None,
        }
        metadata["preview"] = (
            "actual_output_scale_source_with_overlap_feather_ramps_crop_outlines_and_tile_ids"
        )
        metadata_json = json.dumps(metadata, indent=2)

        # Exact output-space masks replace the source-scale ones for everything
        # downstream: stitching here, and the Finalizer's blend later.
        masks = _output_scale_blend_masks(
            metadata, device=upscaled_tiles.device, dtype=upscaled_tiles.dtype
        )
        upscaled_image = SmartTileBlender().blend(
            upscaled_tiles,
            metadata_json,
            masks,
            scale_mode="metadata_scale_factor",
        )[0]
        processing_preview = _draw_adaptive_processing_preview(
            upscaled_image[0],
            [tile for tile in metadata["tiles"] if tile["source_index"] == 0],
            scale_factor,
            overlap,
            int(metadata["output_width"]),
            int(metadata["output_height"]),
            output_feather=int(feather),
            max_preview_size=int(preview_max_size),
        )

        tile_mebibytes = (
            upscaled_tiles.numel() * upscaled_tiles.element_size() / (1024.0 * 1024.0)
        )
        preflight_summary = (
            f"Output: {int(metadata['output_width'])} x {int(metadata['output_height'])}\n"
            f"Grid: {int(metadata['grid_columns'])} columns x {int(metadata['grid_rows'])} rows "
            f"= {int(metadata['tile_count'])} tiles\n"
            f"Sampler tiles: {target_width} x {target_height} | divisible by {int(divisible_by)}\n"
            f"Overlap: {int(overlap)} px | Feather: {int(feather)} px\n"
            f"Scale: {float(scale_factor):g}x | Enlargement: {str(upscale_method)}"
            f"{f' | model native {float(getattr(upscale_model, 'scale', 1.0)):g}x' if str(upscale_method) == UPSCALE_METHODS[0] else ''}\n"
            f"Persistent preprocessing cache: {cache_status}"
            f"{f' | {cache_key[:12]}' if cache_key else ''}\n"
            f"Active upscale batch: {int(upscale_batch_size)} tile(s) (lower this first if VRAM is tight)\n"
            f"Stored tile tensor: approximately {tile_mebibytes:.1f} MiB "
            "(ComfyUI normally keeps this outside active model VRAM)"
        )
        return (
            upscaled_tiles,
            processing_preview.unsqueeze(0),
            upscaled_image,
            masks,
            metadata_json,
            preflight_summary,
        )
