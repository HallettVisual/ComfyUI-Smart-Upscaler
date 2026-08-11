import torch
import torch.nn.functional as F


_STRUCTURE_LONG_SIDE = {
    "conservative": 64,
    "balanced": 96,
    "permissive": 160,
}

_DETAIL_THRESHOLDS = {
    "conservative": 0.040,
    "balanced": 0.022,
    "permissive": 0.010,
}


def _resize_like(image, reference):
    if image.shape[1:3] == reference.shape[1:3]:
        return image
    return F.interpolate(
        image.movedim(-1, 1),
        size=reference.shape[1:3],
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).movedim(1, -1)


def _low_frequency(image, long_side):
    height, width = image.shape[1:3]
    scale = min(1.0, float(long_side) / max(height, width))
    low_height = max(1, round(height * scale))
    low_width = max(1, round(width * scale))
    channels_first = image.movedim(-1, 1)
    reduced = F.interpolate(channels_first, size=(low_height, low_width), mode="area")
    return F.interpolate(
        reduced,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).movedim(1, -1)


def _structure_guard(generated, source, detail_support, reference_mode):
    source_low = _low_frequency(source, _STRUCTURE_LONG_SIDE[detail_support])
    generated_low = _low_frequency(generated, _STRUCTURE_LONG_SIDE[detail_support])
    source_detail = source - source_low
    generated_detail = generated - generated_low

    edge_energy = source_detail.abs().mean(dim=-1, keepdim=True)
    support = (edge_energy / _DETAIL_THRESHOLDS[detail_support]).clamp(0.0, 1.0)
    support = F.avg_pool2d(
        support.movedim(-1, 1), kernel_size=7, stride=1, padding=3
    ).movedim(1, -1)
    if reference_mode == "structure_only":
        source_shape = source_detail.mean(dim=-1, keepdim=True).expand_as(source_detail)
        supported_detail = source_shape * (1.0 - support) + generated_detail * support
        base = generated_low
    else:
        supported_detail = source_detail * (1.0 - support) + generated_detail * support
        base = source_low
    return (base + supported_detail).clamp(0.0, 1.0)


def _channel_stats(image):
    mean = image.mean(dim=(1, 2), keepdim=True)
    std = image.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-4)
    return mean, std


def _color_match(image, source, method):
    if method == "none":
        return image
    if method == "local_tone":
        # Match the source's brightness region by region (low frequency only),
        # fixing local tone drift while preserving generated color and detail.
        weights = image.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 1, 1, 3)
        image_luma = (image * weights).sum(dim=-1, keepdim=True)
        source_luma = (source * weights).sum(dim=-1, keepdim=True)
        delta = _low_frequency(source_luma, 96) - _low_frequency(image_luma, 96)
        return image + delta
    if method == "rgb_mean":
        image_mean, _ = _channel_stats(image)
        source_mean, _ = _channel_stats(source)
        return image + source_mean - image_mean
    if method == "rgb_mean_std":
        image_mean, image_std = _channel_stats(image)
        source_mean, source_std = _channel_stats(source)
        scale = (source_std / image_std).clamp(0.5, 2.0)
        return (image - image_mean) * scale + source_mean

    weights = image.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 1, 1, 3)
    image_luma = (image * weights).sum(dim=-1, keepdim=True)
    source_luma = (source * weights).sum(dim=-1, keepdim=True)
    image_mean, image_std = _channel_stats(image_luma)
    source_mean, source_std = _channel_stats(source_luma)
    matched_luma = (image_luma - image_mean) * (source_std / image_std).clamp(0.5, 2.0)
    matched_luma = matched_luma + source_mean
    return image + matched_luma - image_luma


def _prepare_source(generated_tile, source_tile):
    source = _resize_like(source_tile, generated_tile).to(
        device=generated_tile.device,
        dtype=generated_tile.dtype,
    )
    if source.shape[0] == 1 and generated_tile.shape[0] > 1:
        source = source.expand(generated_tile.shape[0], -1, -1, -1)
    if source.shape != generated_tile.shape:
        raise ValueError("Source and generated tile batches must have matching shapes.")
    return source


# Quick-correction presets: each maps to exact dial values. "Manual" leaves the
# dials in charge, so existing workflows behave identically.
COLOR_MATCH_PRESETS = {
    "Manual (use dials below)": None,
    "No color change (style/lighting edits)": ("none", 0),
    "Match source brightness (recommended)": ("luminance", 50),
    "Even out local brightness": ("local_tone", 60),
    "Match source colors fully": ("rgb_mean_std", 60),
}


class SmartTileColorMatch:
    CATEGORY = "Smart Upscaler/Processing"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("color_matched_tile",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_tile": ("IMAGE",),
                "source_tile": ("IMAGE",),
                "color_match_method": (
                    ["none", "luminance", "local_tone", "rgb_mean", "rgb_mean_std"],
                    {
                        "default": "none",
                        "label": "Match Colors to the Original",
                        "tooltip": "Off by default. none: keep generated colors. luminance: match the original's overall brightness (safest). local_tone: even out brightness region by region, keeping generated color and detail. rgb_mean: match average color. rgb_mean_std: match color and contrast fully.",
                    },
                ),
                "color_match_strength": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                        "tooltip": "0 keeps the generated tile untouched; 100 fully applies the chosen match to the source.",
                    },
                ),
            },
            "optional": {
                "color_preset": (
                    list(COLOR_MATCH_PRESETS),
                    {
                        "default": "Manual (use dials below)",
                        "label": "Quick Preset",
                        "tooltip": "One-click settings that fill in the two dials above and stay shown as a label. Pick No color change for day-to-night or style edits, Match source brightness for faithful photo upscales. Fine-tune the dials freely afterward.",
                    },
                ),
            },
        }

    def apply(
        self,
        generated_tile,
        source_tile,
        color_match_method,
        color_match_strength,
        color_preset="Manual (use dials below)",
    ):
        # The preset dropdown writes values into the visible dials (see
        # web/finishing_presets.js) and stays selected as a label; the dials are
        # always the source of truth here.
        source = _prepare_source(generated_tile, source_tile)
        strength = float(color_match_strength) / 100.0
        if color_match_method == "none" or strength <= 0.0:
            return (generated_tile.clamp(0.0, 1.0),)
        matched = _color_match(generated_tile, source, color_match_method).clamp(0.0, 1.0)
        return (torch.lerp(generated_tile, matched, strength).clamp(0.0, 1.0),)


class SmartTileFidelityColorMatch:
    CATEGORY = "Smart Upscaler/Processing"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("corrected_tile",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_tile": ("IMAGE",),
                "source_tile": ("IMAGE",),
                "reference_mode": (
                    ["appearance_and_structure", "structure_only"],
                    {"default": "appearance_and_structure"},
                ),
                "structure_preservation": (
                    "INT",
                    {"default": 55, "min": 0, "max": 100, "step": 1},
                ),
                "detail_support": (
                    ["conservative", "balanced", "permissive"],
                    {"default": "balanced"},
                ),
                "color_match_method": (
                    ["none", "luminance", "local_tone", "rgb_mean", "rgb_mean_std"],
                    {"default": "luminance"},
                ),
                "color_match_strength": (
                    "INT",
                    {"default": 50, "min": 0, "max": 100, "step": 1},
                ),
            },
        }

    def apply(
        self,
        generated_tile,
        source_tile,
        structure_preservation,
        detail_support,
        color_match_method,
        color_match_strength,
        reference_mode="appearance_and_structure",
    ):
        source = _prepare_source(generated_tile, source_tile)

        output = generated_tile
        structure_strength = float(structure_preservation) / 100.0
        if structure_strength > 0.0:
            guarded = _structure_guard(output, source, detail_support, reference_mode)
            output = torch.lerp(output, guarded, structure_strength)

        color_strength = float(color_match_strength) / 100.0
        if color_match_method != "none" and color_strength > 0.0:
            matched = _color_match(output, source, color_match_method).clamp(0.0, 1.0)
            output = torch.lerp(output, matched, color_strength)

        return (output.clamp(0.0, 1.0),)
