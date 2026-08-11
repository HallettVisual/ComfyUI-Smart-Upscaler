"""Stitching coverage must be gapless at every scale.

Regression tests for two real failures:
1. 6x scale: masks resized from source scale drifted whenever the output tile
   was not an exact multiple of the padded source tile, pushing mask zeros into
   covered pixels ("Blend masks leave uncovered pixels in the stitched image").
2. Fractional scales (2.5x) on odd image sizes: independently rounding tile
   position and width opened 1px gaps between neighboring tiles.
"""

import json
import unittest

import torch

from nodes.blending import SmartTileBlender
from nodes.tiling import SmartAdaptiveTilePlanner
from nodes.upscaled_tiling import _output_scale_blend_masks


def _coverage_holes(width, height, scale, overlap=64, feather=32):
    image = torch.zeros(1, height, width, 3)
    _, _, _, metadata_json = SmartAdaptiveTilePlanner().plan(
        image, 1024, 1536, overlap, feather, float(scale), 16, "edge"
    )
    metadata = json.loads(metadata_json)
    scale_value = float(metadata["scale_factor"])
    canvas_width = int(
        metadata.get("output_width", round(metadata["image_width"] * scale_value))
    )
    canvas_height = int(
        metadata.get("output_height", round(metadata["image_height"] * scale_value))
    )
    masks = _output_scale_blend_masks(metadata, device="cpu", dtype=torch.float32)
    weight = torch.zeros(canvas_height, canvas_width)
    for record in metadata["tiles"]:
        index = int(record["tile_index"])
        x0 = round(record["x"] * scale_value)
        y0 = round(record["y"] * scale_value)
        x1 = min(canvas_width, max(x0 + 1, round((record["x"] + record["width"]) * scale_value)))
        y1 = min(canvas_height, max(y0 + 1, round((record["y"] + record["height"]) * scale_value)))
        weight[y0:y1, x0:x1] += masks[index][: y1 - y0, : x1 - x0]
    return int((weight <= 1e-6).sum())


class TileCoverageTests(unittest.TestCase):
    def test_six_x_scale_has_no_uncovered_pixels(self):
        # The exact reported failure: 1340x896 at 6x, overlap 64, feather 32.
        self.assertEqual(_coverage_holes(1340, 896, 6), 0)

    def test_fractional_scale_on_odd_sizes_has_no_gaps(self):
        self.assertEqual(_coverage_holes(1343, 897, 2.5), 0)
        self.assertEqual(_coverage_holes(1155, 863, 2.5), 0)

    def test_twelve_x_scale_is_supported_and_gapless(self):
        self.assertEqual(_coverage_holes(1340, 896, 12), 0)

    def test_no_overlap_and_no_feather_still_covers(self):
        self.assertEqual(_coverage_holes(1340, 896, 6, overlap=0, feather=0), 0)

    def test_full_blend_runs_end_to_end_at_six_x(self):
        image = torch.rand(1, 896, 1340, 3)
        tiles, _, _, metadata_json = SmartAdaptiveTilePlanner().plan(
            image, 1024, 1536, 64, 32, 6.0, 16, "edge"
        )
        metadata = json.loads(metadata_json)
        out_h = int(metadata["output_tile_height"])
        out_w = int(metadata["output_tile_width"])
        resized = torch.nn.functional.interpolate(
            tiles.movedim(-1, 1), size=(out_h, out_w), mode="bilinear",
            align_corners=False,
        ).movedim(1, -1).clamp(0, 1)
        metadata["tiles_are_output_scale"] = True
        masks = _output_scale_blend_masks(metadata, device="cpu", dtype=torch.float32)
        stitched = SmartTileBlender().blend(
            resized, json.dumps(metadata), masks, scale_mode="metadata_scale_factor"
        )[0]
        self.assertEqual(
            stitched.shape[1:3],
            (round(896 * 6.0), round(1340 * 6.0)),
        )


class StitchMemoryGuardTests(unittest.TestCase):
    def test_impossible_canvas_gets_a_plain_answer_not_a_crash(self):
        from nodes.blending import _stitch_memory_guard

        # A 4K source at 9x (the reported case) against 8 GB free memory.
        with self.assertRaises(ValueError) as context:
            _stitch_memory_guard(
                3840 * 9, 2160 * 9, 3, 1, available_bytes=8 * 1024**3
            )
        message = str(context.exception)
        self.assertIn("gigapixels", message)
        self.assertIn("Lower the scale factor", message)

    def test_reasonable_canvas_passes(self):
        from nodes.blending import _stitch_memory_guard

        _stitch_memory_guard(8040, 5376, 3, 1, available_bytes=8 * 1024**3)


if __name__ == "__main__":
    unittest.main()
