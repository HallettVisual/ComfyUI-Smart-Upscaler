import hashlib
import json

import torch
import torch.nn.functional as F


def _single_metadata(metadata_values):
    values = metadata_values if isinstance(metadata_values, list) else [metadata_values]
    if not values:
        raise ValueError("Tile metadata is required.")
    try:
        return json.loads(values[0])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Tile metadata must be valid JSON from Smart Tile Planner.") from exc


def _flatten_image_values(values):
    flattened = []
    pending = list(values if isinstance(values, list) else [values])
    while pending:
        value = pending.pop(0)
        if isinstance(value, list):
            pending[0:0] = value
        else:
            flattened.append(value)
    return flattened


def _tile_image_list(values):
    """Normalize either a Comfy IMAGE batch or a list of one-image batches."""
    tiles = []
    for value in _flatten_image_values(values):
        if not isinstance(value, torch.Tensor) or value.ndim != 4:
            raise ValueError("Sampler tile images must come from Tile Job Director.")
        tiles.extend(value[index : index + 1] for index in range(int(value.shape[0])))
    return tiles


class SmartSamplerTileSelector:
    """Select one complete sampler job after all tile prompts have been generated.

    The selector deliberately operates on every sampler-facing field together.
    Selecting only the IMAGE path would let ComfyUI pair one tile with the wrong
    prompt or execute it repeatedly for every remaining conditioning item.
    """

    CATEGORY = "Smart Upscaler/Processing"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = (
        "tile_images",
        "positive_prompts",
        "negative_prompts",
        "tile_references",
        "tile_seeds",
    )
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True, True, True, True)
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_images": ("IMAGE", {"forceInput": True}),
                "positive_prompts": ("STRING", {"forceInput": True}),
                "negative_prompts": ("STRING", {"forceInput": True}),
                "tile_references": ("STRING", {"forceInput": True}),
                "tile_seeds": ("INT", {"forceInput": True}),
                "processing_mode": (
                    ["All tiles (production)", "One tile test"],
                    {
                        "default": "All tiles (production)",
                        "label": "Sampler Processing",
                        "tooltip": (
                            "All tiles runs the normal workflow. One tile test keeps "
                            "all upstream prompt generation and audit records, but sends "
                            "only the selected aligned tile job to VAE encode and sampling."
                        ),
                    },
                ),
                "tile_number": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                        "label": "Test Tile Number (T-number)",
                        "tooltip": (
                            "The human tile number shown on the processing grid and "
                            "prompt inspector. Used only in One tile test mode."
                        ),
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, processing_mode=None):
        # Keep loading safe if labels are refined in a later UI release.
        return True

    def select(
        self,
        tile_images,
        positive_prompts,
        negative_prompts,
        tile_references,
        tile_seeds,
        processing_mode,
        tile_number,
    ):
        images = _tile_image_list(tile_images)
        positives = [str(value) for value in _flatten_image_values(positive_prompts)]
        negatives = [str(value) for value in _flatten_image_values(negative_prompts)]
        references = [str(value) for value in _flatten_image_values(tile_references)]
        seeds = [int(value) for value in _flatten_image_values(tile_seeds)]
        counts = {
            "tile images": len(images),
            "positive prompts": len(positives),
            "negative prompts": len(negatives),
            "tile references": len(references),
            "tile seeds": len(seeds),
        }
        expected = len(references)
        if expected == 0:
            raise ValueError("Sampler Tile Selector received no tile jobs.")
        mismatched = [
            f"{name}={count}" for name, count in counts.items() if count != expected
        ]
        if mismatched:
            raise ValueError(
                "Sampler Tile Selector inputs are not aligned: "
                + ", ".join(mismatched)
                + f"; expected {expected} of each. Connect all five inputs from the "
                "same Tile Job Director and Exact-Tile Prompt path."
            )

        parsed_references = []
        seen_indexes = set()
        for reference_text in references:
            try:
                reference = json.loads(reference_text)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "Sampler Tile Selector references must come from Exact-Tile Prompt."
                ) from exc
            tile_index = int(reference.get("tile_index", -1))
            if tile_index < 0:
                raise ValueError("Sampler Tile Selector received a reference without tile_index.")
            if tile_index in seen_indexes:
                raise ValueError(
                    f"Sampler Tile Selector received tile index {tile_index} more than once."
                )
            seen_indexes.add(tile_index)
            parsed_references.append(reference)

        mode_values = _flatten_image_values(processing_mode)
        mode = str(mode_values[0] if mode_values else "All tiles (production)")
        if "one tile" not in mode.casefold() and "single" not in mode.casefold():
            return images, positives, negatives, references, seeds

        number_values = _flatten_image_values(tile_number)
        selected_number = int(number_values[0] if number_values else 1)
        selected_position = next(
            (
                position
                for position, reference in enumerate(parsed_references)
                if int(reference["tile_index"]) + 1 == selected_number
            ),
            None,
        )
        if selected_position is None:
            available = sorted(int(reference["tile_index"]) + 1 for reference in parsed_references)
            if available:
                available_text = (
                    f"T{available[0]:03d}-T{available[-1]:03d}"
                    if available == list(range(available[0], available[-1] + 1))
                    else ", ".join(f"T{value:03d}" for value in available[:20])
                )
            else:
                available_text = "none"
            raise ValueError(
                f"Test tile T{selected_number:03d} is not available. "
                f"Available tile numbers: {available_text}."
            )

        position = selected_position
        return (
            [images[position]],
            [positives[position]],
            [negatives[position]],
            [references[position]],
            [seeds[position]],
        )


class SmartTileMergePartialBatch:
    CATEGORY = "Smart Upscaler/Processing"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("tile_batch",)
    INPUT_IS_LIST = True
    FUNCTION = "merge"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_images": ("IMAGE", {"forceInput": True}),
                "tile_references": ("STRING", {"forceInput": True}),
                "source_tiles": ("IMAGE", {"forceInput": True}),
                "tile_metadata_json": ("STRING", {"forceInput": True}),
                "fallback_method": (
                    ["bilinear", "bicubic", "area", "nearest-exact"],
                    {"default": "bilinear"},
                ),
            },
        }

    def merge(
        self,
        processed_images,
        tile_references,
        source_tiles,
        tile_metadata_json,
        fallback_method,
    ):
        metadata = _single_metadata(tile_metadata_json)
        processed = _flatten_image_values(processed_images)
        references = _flatten_image_values(tile_references)
        source_values = _flatten_image_values(source_tiles)
        method_values = _flatten_image_values(fallback_method)
        method = str(method_values[0] if method_values else "bilinear")
        if not source_values or not isinstance(source_values[0], torch.Tensor):
            raise ValueError("Source tiles must come from a Smart tile planner.")
        source_batch = source_values[0]
        expected_count = len(metadata.get("tiles", []))
        if source_batch.shape[0] != expected_count:
            raise ValueError(
                f"Source tile batch contains {source_batch.shape[0]} tiles, but metadata expects {expected_count}."
            )
        if len(processed) != len(references):
            raise ValueError("Processed tile images and references must have matching counts.")

        target_width = int(
            metadata.get(
                "output_tile_width",
                round(metadata["tile_width"] * float(metadata.get("scale_factor", 1.0))),
            )
        )
        target_height = int(
            metadata.get(
                "output_tile_height",
                round(metadata["tile_height"] * float(metadata.get("scale_factor", 1.0))),
            )
        )
        kwargs = {}
        if method in ("bilinear", "bicubic"):
            kwargs = {"align_corners": False, "antialias": True}
        baseline = F.interpolate(
            source_batch.movedim(-1, 1),
            size=(target_height, target_width),
            mode=method,
            **kwargs,
        ).movedim(1, -1)

        seen_indexes = set()
        for processed_image, reference_json in zip(processed, references):
            try:
                reference = json.loads(reference_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("Tile references must come from Smart Tile Prompt Batch.") from exc
            tile_index = int(reference.get("tile_index", -1))
            if not 0 <= tile_index < expected_count:
                raise ValueError(f"Processed tile has invalid index {tile_index}.")
            if tile_index in seen_indexes:
                raise ValueError(f"Processed tile index {tile_index} was received more than once.")
            if not isinstance(processed_image, torch.Tensor) or processed_image.shape != (
                1,
                target_height,
                target_width,
                source_batch.shape[3],
            ):
                received = (
                    "x".join(str(v) for v in processed_image.shape)
                    if isinstance(processed_image, torch.Tensor)
                    else type(processed_image).__name__
                )
                raise ValueError(
                    f"Processed tile {tile_index} must be "
                    f"1x{target_height}x{target_width}x{source_batch.shape[3]} "
                    f"(height x width), but received {received}. If the two sizes are "
                    "swapped or rounded, check that the sampler's latent width and "
                    "height come from this exact tile image and are not rounded by "
                    "the model's latent size rules."
                )
            baseline[tile_index] = processed_image[0].to(
                device=baseline.device,
                dtype=baseline.dtype,
            )
            seen_indexes.add(tile_index)

        return (baseline.clamp(0.0, 1.0),)


class SmartTileSeed:
    CATEGORY = "Smart Upscaler/Processing"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("tile_seed",)
    FUNCTION = "derive"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_reference": ("STRING", {"forceInput": True}),
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

    def derive(self, tile_reference, base_seed, seed_mode):
        try:
            reference = json.loads(tile_reference)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Tile reference must come from Smart Tile Prompt Batch.") from exc

        tile_index = int(reference.get("tile_index", 0))
        source_index = int(reference.get("source_index", 0))
        if seed_mode == "fixed":
            seed = base_seed
        elif seed_mode == "increment":
            seed = base_seed + tile_index
        else:
            digest = hashlib.sha256(f"{base_seed}:{source_index}:{tile_index}".encode("ascii")).digest()
            seed = int.from_bytes(digest[:8], "big")
        return (seed & 0xFFFFFFFFFFFFFFFF,)
