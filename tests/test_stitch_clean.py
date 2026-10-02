"""Stitch Tiles - Clean: tiles go back together without the original's detail."""

import json
import unittest
from unittest.mock import patch

import torch

from nodes.fidelity import _low_frequency
from nodes.stitch import COLOR_CHOICES, JOIN_CHOICES, SmartTileStitchClean, _color_plan
from nodes.upscaled_tiling import SmartUpscaledTilePlanner


class StitchCleanTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.source = torch.rand(1, 40, 56, 3) * 0.3 + 0.35

        def resize(source_tiles, target_width, target_height, method):
            return torch.nn.functional.interpolate(
                source_tiles.movedim(-1, 1), size=(target_height, target_width), mode="bilinear"
            ).movedim(1, -1)

        with patch("nodes.upscaled_tiling._resize_tiles_without_model", side_effect=resize):
            (
                self.tiles,
                _,
                _,
                self.masks,
                self.metadata_json,
                _,
            ) = SmartUpscaledTilePlanner().plan_and_upscale(
                self.source, 32, 48, 8, 4, 2.0, 8, 1, "edge", "bypass", "auto", 512,
                upscale_method="Bilinear (gentle standard resize)",
            )
        self.metadata = json.loads(self.metadata_json)
        self.references = [
            json.dumps({"tile_index": index, "tile_id": f"T{index + 1:03d}"})
            for index in range(self.tiles.shape[0])
        ]
        # The model drew new fine detail and tinted every tile differently.
        detail = torch.rand(self.tiles.shape) * 0.08
        tints = torch.linspace(-0.12, 0.12, self.tiles.shape[0]).view(-1, 1, 1, 1)
        self.generated = (self.tiles + detail + tints).clamp(0.0, 1.0)

    def _stitch(self, colors, joins, processed=None, references=None, prompt_system=None):
        processed = processed if processed is not None else [
            self.generated[index : index + 1] for index in range(self.generated.shape[0])
        ]
        arguments = [
            processed,
            references or self.references,
            [self.tiles],
            [self.metadata_json],
            [self.masks],
            [colors],
            [joins],
        ]
        if prompt_system is not None:
            return SmartTileStitchClean().stitch(*arguments, [prompt_system])
        return SmartTileStitchClean().stitch(*arguments)

    def test_stitch_only_places_every_tile_exactly_as_drawn(self):
        self.assertGreater(self.tiles.shape[0], 1)
        _, final = self._stitch(COLOR_CHOICES[3], JOIN_CHOICES[1])
        scale = float(self.metadata["scale_factor"])
        for record in self.metadata["tiles"]:
            x0 = round(record["x"] * scale)
            y0 = round(record["y"] * scale)
            cx0 = round(record["core_x"] * scale)
            cy0 = round(record["core_y"] * scale)
            cx1 = round((record["core_x"] + record["core_width"]) * scale)
            cy1 = round((record["core_y"] + record["core_height"]) * scale)
            tile = self.generated[int(record["tile_index"])]
            self.assertTrue(
                torch.allclose(
                    final[0, cy0:cy1, cx0:cx1],
                    tile[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0],
                    atol=1e-6,
                )
            )

    def test_matching_colors_never_touches_fine_detail(self):
        corrected, _ = self._stitch(COLOR_CHOICES[1], JOIN_CHOICES[0])
        before_spread = []
        after_spread = []
        for index, tile in enumerate(corrected):
            generated = self.generated[index : index + 1]
            # Fine detail (above the broad-colour scale) is the model's, unchanged.
            fine = lambda image: image - _low_frequency(image, 32)
            self.assertLess(float((fine(tile) - fine(generated)).abs().mean()), 0.01)
            before_spread.append(float((generated - self.tiles[index]).mean()))
            after_spread.append(float((tile - self.tiles[index]).mean()))
        # The per-tile tints are gone: every tile now sits on the original's colour.
        self.assertGreater(max(before_spread) - min(before_spread), 0.2)
        self.assertLess(max(after_spread) - min(after_spread), 0.02)

    def test_automatic_follows_the_prompt_director(self):
        self.assertEqual(_color_plan(COLOR_CHOICES[0], None), "original")
        self.assertEqual(_color_plan(COLOR_CHOICES[0], {"operation_mode": "detail_enhance"}), "original")
        self.assertEqual(_color_plan(COLOR_CHOICES[0], {"operation_mode": "style_transform"}), "even")
        restyled, _ = self._stitch(
            COLOR_CHOICES[0], JOIN_CHOICES[0], prompt_system={"operation_mode": "style_transform"}
        )
        evened, _ = self._stitch(COLOR_CHOICES[2], JOIN_CHOICES[0])
        for first, second in zip(restyled, evened):
            self.assertTrue(torch.equal(first, second))

    def test_one_tile_test_and_rgba_tiles_still_stitch(self):
        rgba = torch.cat(
            (self.generated[1:2], torch.ones(1, *self.generated.shape[1:3], 1)), -1
        )
        corrected, final = self._stitch(
            COLOR_CHOICES[0], JOIN_CHOICES[0], processed=[rgba], references=[self.references[1]]
        )
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0].shape[-1], 3)
        self.assertEqual(final.shape[-1], 3)


if __name__ == "__main__":
    unittest.main()
