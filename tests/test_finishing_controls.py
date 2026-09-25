"""Quick presets and the local_tone method on the finishing nodes.

These additions must be strictly non-destructive: with the preset on "Manual",
both nodes behave exactly as before the controls existed.
"""

import json
import unittest

import torch

from nodes.fidelity import (
    COLOR_MATCH_METHODS,
    COLOR_MATCH_PRESETS,
    SmartTileColorMatch,
    _color_match,
)
from nodes.finalize import (
    FINISH_PRESETS,
    SmartTileFinalizer,
    _cross_tile_overlap_consistency,
)
from nodes.tiling import SmartTilePlanner


class ColorMatchPresetTests(unittest.TestCase):
    def test_manual_preset_changes_nothing(self):
        torch.manual_seed(1)
        generated = torch.rand(1, 16, 16, 3)
        source = torch.rand(1, 16, 16, 3)
        baseline = SmartTileColorMatch().apply(generated, source, "luminance", 50)[0]
        manual = SmartTileColorMatch().apply(
            generated, source, "luminance", 50, color_preset="Manual (use dials below)"
        )[0]
        self.assertTrue(torch.equal(baseline, manual))

    def test_preset_is_a_label_only_and_the_dials_always_rule(self):
        # The preset dropdown writes values into the visible widgets in the UI
        # and stays selected as a label; server-side it never overrides dials.
        torch.manual_seed(2)
        generated = torch.rand(1, 16, 16, 3)
        source = torch.rand(1, 16, 16, 3)
        with_preset = SmartTileColorMatch().apply(
            generated,
            source,
            "rgb_mean_std",
            100,
            color_preset="No color change (style/lighting edits)",
        )[0]
        without_preset = SmartTileColorMatch().apply(
            generated, source, "rgb_mean_std", 100
        )[0]
        self.assertTrue(torch.equal(with_preset, without_preset))

    def test_every_preset_maps_to_valid_dial_values(self):
        methods = SmartTileColorMatch.INPUT_TYPES()["required"]["color_match_method"][0]
        self.assertEqual(methods, list(COLOR_MATCH_METHODS))
        for name, mapping in COLOR_MATCH_PRESETS.items():
            if mapping is None:
                continue
            method, strength = mapping
            self.assertIn(method, methods, name)
            self.assertTrue(0 <= strength <= 100, name)


class OriginalColorsMethodTests(unittest.TestCase):
    def _neighbours(self):
        # One source strip cut into two tiles; the model tinted each tile
        # differently, the classic checkerboard cast.
        torch.manual_seed(4)
        source = torch.rand(1, 64, 64, 3) * 0.2 + 0.4
        detail = torch.rand(1, 64, 64, 3) * 0.1
        left = (source + detail + torch.tensor([0.12, 0.0, -0.08])).clamp(0, 1)
        right = (source + detail + torch.tensor([-0.10, 0.06, 0.05])).clamp(0, 1)
        return source, left, right

    def test_differently_tinted_tiles_come_back_the_same_color(self):
        source, left, right = self._neighbours()
        node = SmartTileColorMatch()
        fixed_left = node.apply(left, source, "original_colors", 100)[0]
        fixed_right = node.apply(right, source, "original_colors", 100)[0]
        before = (left.mean(dim=(1, 2)) - right.mean(dim=(1, 2))).abs().max()
        after = (fixed_left.mean(dim=(1, 2)) - fixed_right.mean(dim=(1, 2))).abs().max()
        self.assertGreater(float(before), 0.1)
        self.assertLess(float(after), 0.01)
        # The model's fine detail survives: only broad color was replaced.
        self.assertGreater(float((fixed_left - source).std()), 0.02)

    def test_automatic_leaves_a_look_change_alone(self):
        source, left, _ = self._neighbours()
        node = SmartTileColorMatch()
        restyled = node.apply(
            left, source, "automatic", 100, prompt_system={"operation_mode": "style_transform"}
        )[0]
        self.assertTrue(torch.equal(restyled, left.clamp(0, 1)))
        faithful = node.apply(
            left, source, "automatic", 100, prompt_system={"operation_mode": "detail_enhance"}
        )[0]
        expected = node.apply(left, source, "original_colors", 100)[0]
        self.assertTrue(torch.equal(faithful, expected))
        # Not wired to the Prompt Director: keep the original colors.
        self.assertTrue(torch.equal(node.apply(left, source, "automatic", 100)[0], expected))

    def test_a_retired_preset_name_in_a_saved_graph_still_queues(self):
        self.assertTrue(
            SmartTileColorMatch.VALIDATE_INPUTS(color_preset="Match source brightness (recommended)")
        )


class LocalToneMethodTests(unittest.TestCase):
    def test_local_tone_follows_source_brightness_regionally(self):
        # Source: bright left half, dark right half. Generated: uniform mid gray.
        source = torch.full((1, 32, 32, 3), 0.5)
        source[:, :, :16] = 0.9
        source[:, :, 16:] = 0.1
        generated = torch.full((1, 32, 32, 3), 0.5)
        matched = _color_match(generated, source, "local_tone")
        left = float(matched[:, :, :12].mean())
        right = float(matched[:, :, 20:].mean())
        self.assertGreater(left, right + 0.3)

    def test_local_tone_preserves_generated_hue(self):
        # Generated is reddish; source is neutral gray. Brightness may shift but
        # the red-versus-green relationship must survive.
        generated = torch.zeros(1, 24, 24, 3)
        generated[..., 0] = 0.7
        generated[..., 1] = 0.3
        generated[..., 2] = 0.3
        source = torch.full((1, 24, 24, 3), 0.5)
        matched = _color_match(generated, source, "local_tone")
        red_minus_green = float((matched[..., 0] - matched[..., 1]).mean())
        self.assertGreater(red_minus_green, 0.35)


class FinalizerPresetTests(unittest.TestCase):
    def _run(self, **overrides):
        torch.manual_seed(3)
        image = torch.rand(1, 8, 12, 3)
        tiles, _, masks, metadata_json = SmartTilePlanner().plan(
            image,
            tile_width=6,
            tile_height=8,
            overlap=2,
            feather=1,
            scale_factor=1.0,
            padding_mode="edge",
        )
        references = [
            json.dumps({"tile_index": index, "source_index": 0})
            for index in range(tiles.shape[0])
        ]
        processed = [tiles[index : index + 1] * 0.9 for index in range(tiles.shape[0])]
        arguments = {
            "processed_images": processed,
            "tile_references": references,
            "source_tiles": [tiles],
            "tile_metadata_json": [metadata_json],
            "blend_masks": [masks],
            "reference_mode": ["Match original appearance + structure (normal upscale)"],
            "structure_preservation": [55],
            "detail_support": ["Balanced"],
            "fallback_method": ["bilinear"],
            "cross_tile_consistency": [35],
        }
        arguments.update(overrides)
        return SmartTileFinalizer().finalize(**arguments)

    def test_rgba_tiles_from_qwen_image_21_are_accepted(self):
        # Qwen Image 2.1's VAE decodes RGBA; the alpha of an opaque upscale is 1.
        torch.manual_seed(3)
        image = torch.rand(1, 8, 12, 3)
        tiles = SmartTilePlanner().plan(
            image, tile_width=6, tile_height=8, overlap=2, feather=1,
            scale_factor=1.0, padding_mode="edge",
        )[0]
        rgba = [
            torch.cat((tiles[index : index + 1] * 0.9, torch.ones(1, *tiles.shape[1:3], 1)), -1)
            for index in range(tiles.shape[0])
        ]
        rgb_tiles, rgb_image = self._run()
        rgba_tiles, rgba_image = self._run(processed_images=rgba)
        self.assertTrue(torch.allclose(rgb_image, rgba_image, atol=1e-6))
        self.assertEqual(rgba_tiles[0].shape[-1], 3)
        matched = SmartTileColorMatch().apply(rgba[0], tiles[0:1], "original_colors", 100)[0]
        self.assertEqual(matched.shape[-1], 3)

    def test_manual_preset_matches_previous_behavior(self):
        baseline_tiles, baseline_image = self._run()
        manual_tiles, manual_image = self._run(
            finish_preset=["Manual (use dials below)"]
        )
        self.assertTrue(torch.equal(baseline_image, manual_image))
        for base, manual in zip(baseline_tiles, manual_tiles):
            self.assertTrue(torch.equal(base, manual))

    def test_finish_preset_is_a_label_and_dials_always_rule(self):
        # A preset name arriving server-side never overrides the dial values
        # (the UI already wrote the preset's values into the dials).
        preset_tiles, preset_image = self._run(
            reference_mode=["Keep generated appearance; guide structure only"],
            structure_preservation=[20],
            detail_support=["Balanced"],
            finish_preset=["Faithful photo upscale"],
        )
        expected_tiles, expected_image = self._run(
            reference_mode=["Keep generated appearance; guide structure only"],
            structure_preservation=[20],
            detail_support=["Balanced"],
        )
        self.assertTrue(torch.allclose(preset_image, expected_image))

    def test_every_finish_preset_uses_valid_choices(self):
        from nodes.finalize import CONSISTENCY_MODES, DETAIL_FREEDOM, REFERENCE_BEHAVIORS

        for name, mapping in FINISH_PRESETS.items():
            if mapping is None:
                continue
            self.assertIn(mapping["reference_mode"], REFERENCE_BEHAVIORS, name)
            self.assertIn(mapping["detail_support"], DETAIL_FREEDOM, name)
            self.assertTrue(0 <= mapping["structure_preservation"] <= 100, name)
            self.assertTrue(0 <= mapping["cross_tile_consistency"] <= 100, name)
            if "consistency_mode" in mapping:
                self.assertIn(mapping["consistency_mode"], CONSISTENCY_MODES, name)

    def test_legacy_reference_mode_wording_still_resolves(self):
        from nodes.finalize import REFERENCE_BEHAVIOR_ALIASES

        self.assertEqual(
            REFERENCE_BEHAVIOR_ALIASES["Keep generated appearance; guide structure only"],
            "structure_only",
        )
        self.assertEqual(
            REFERENCE_BEHAVIOR_ALIASES[
                "Match original appearance + structure (normal upscale)"
            ],
            "appearance_and_structure",
        )

    def test_relabelled_choices_keep_every_saved_graph_working(self):
        """The dials were reworded into plain language. A workflow saved with the
        old strings must resolve to exactly the same behaviour, because widget
        values are stored as text."""
        from nodes.finalize import (
            CONSISTENCY_MODE_ALIASES,
            DETAIL_FREEDOM,
            DETAIL_FREEDOM_ALIASES,
        )

        self.assertEqual(CONSISTENCY_MODE_ALIASES["Even shift per tile"], "even")
        self.assertEqual(
            CONSISTENCY_MODE_ALIASES["Smooth gradient (fixes side-to-side mismatch)"],
            "gradient",
        )
        self.assertEqual(
            DETAIL_FREEDOM_ALIASES["Conservative (less generated detail)"], "conservative"
        )
        self.assertEqual(
            DETAIL_FREEDOM_ALIASES["Permissive (more generated detail)"], "permissive"
        )
        # And the new wording maps to the same three engine values.
        self.assertEqual(
            {DETAIL_FREEDOM_ALIASES[name] for name in DETAIL_FREEDOM},
            {"conservative", "balanced", "permissive"},
        )

    def test_old_and_new_wording_produce_identical_results(self):
        """The dials were reworded into plain language. Widget values are stored
        as text, so a workflow saved on the old strings must behave exactly as
        one saved on the new ones."""
        old_tiles, old_image = self._run(
            detail_support=["Permissive (more generated detail)"],
            cross_tile_consistency=[60],
            consistency_mode=["Smooth gradient (fixes side-to-side mismatch)"],
        )
        new_tiles, new_image = self._run(
            detail_support=["Keep more new detail"],
            cross_tile_consistency=[60],
            consistency_mode=["Fade the fix toward the edge that disagrees (best for seams)"],
        )
        self.assertTrue(torch.equal(old_image, new_image))
        for old, new in zip(old_tiles, new_tiles):
            self.assertTrue(torch.equal(old, new))

    def test_the_recommended_preset_is_faithful_and_seam_hiding(self):
        """The one combination people actually want on a correct photograph -
        follow the original AND hide the joins - had to be built by hand before."""
        preset = FINISH_PRESETS["Photo upscale, seams hidden (start here)"]
        self.assertEqual(preset["reference_mode"], "Stay close to the original photo")
        self.assertGreaterEqual(preset["structure_preservation"], 40)
        self.assertGreaterEqual(preset["cross_tile_consistency"], 60)
        self.assertIn("Fade the fix", preset["consistency_mode"])


class PresetFirstDefaultTests(unittest.TestCase):
    """Both finishing nodes now show one dropdown and hide their dials.

    A fresh node must therefore already hold the values its shipped preset
    would write - otherwise the dial the user cannot see disagrees with the
    label they can.
    """

    @staticmethod
    def _defaults(node):
        spec = node.INPUT_TYPES()
        fields = {**spec["required"], **spec.get("optional", {})}
        return {name: value[1] for name, value in fields.items() if len(value) > 1}

    def test_finalizer_defaults_match_its_shipped_preset(self):
        defaults = self._defaults(SmartTileFinalizer)
        shipped = defaults["finish_preset"]["default"]
        self.assertEqual(shipped, "Photo upscale, seams hidden (start here)")
        for name, value in FINISH_PRESETS[shipped].items():
            self.assertEqual(defaults[name]["default"], value, name)
            self.assertTrue(defaults[name].get("advanced"), name)

    def test_color_match_defaults_match_its_shipped_preset(self):
        defaults = self._defaults(SmartTileColorMatch)
        shipped = defaults["color_preset"]["default"]
        self.assertEqual(
            shipped,
            "Automatic - original colors unless the task changes the look (recommended)",
        )
        method, strength = COLOR_MATCH_PRESETS[shipped]
        self.assertEqual(defaults["color_match_method"]["default"], method)
        self.assertEqual(defaults["color_match_strength"]["default"], strength)
        self.assertTrue(defaults["color_match_method"].get("advanced"))
        self.assertTrue(defaults["color_match_strength"].get("advanced"))


class GradientSeamFixTests(unittest.TestCase):
    """A tile that only disagrees with its neighbor on one side cannot be fixed
    by a constant shift; the gradient mode corrects exactly the seam."""

    def _three_tile_setup(self):
        # Three 16-wide tiles in a row, 4px overlaps. The outer tiles are flat
        # mid gray; the middle tile carries an internal left-to-right tint drift
        # (the classic VAE artifact), so it is too dark against its left
        # neighbor and too bright against its right neighbor at the same time.
        # No single constant shift of the middle tile can fix both seams.
        metadata = {
            "scale_factor": 1.0,
            "tiles": [
                {"tile_index": 0, "source_index": 0, "row": 0, "column": 0,
                 "x": 0, "y": 0, "width": 16, "height": 16},
                {"tile_index": 1, "source_index": 0, "row": 0, "column": 1,
                 "x": 12, "y": 0, "width": 16, "height": 16},
                {"tile_index": 2, "source_index": 0, "row": 0, "column": 2,
                 "x": 24, "y": 0, "width": 16, "height": 16},
            ],
        }
        tiles = torch.full((3, 16, 16, 3), 0.5)
        drift = torch.linspace(0.44, 0.56, 16).view(1, 16, 1)
        tiles[1] = drift.expand(16, 16, 3)
        return tiles, metadata

    def _seam_gaps(self, tiles):
        left = float((tiles[0, :, 12:16] - tiles[1, :, 0:4]).abs().mean())
        right = float((tiles[1, :, 12:16] - tiles[2, :, 0:4]).abs().mean())
        return left + right

    def test_gradient_mode_fixes_the_drifting_tile_not_its_neighbors(self):
        tiles, metadata = self._three_tile_setup()
        even = _cross_tile_overlap_consistency(tiles, metadata, 100, "even")
        gradient = _cross_tile_overlap_consistency(tiles, metadata, 100, "gradient")
        # Both modes close the seams comparably...
        self.assertLess(self._seam_gaps(gradient), self._seam_gaps(even) * 1.2)
        self.assertLess(self._seam_gaps(gradient), self._seam_gaps(tiles) * 0.25)
        # ...but even mode drags the CORRECT outer tiles toward the drifting
        # middle tile, while gradient mode corrects the drifting tile itself
        # and leaves correct tiles close to their true color.
        even_collateral = abs(float(even[0].mean()) - 0.5)
        gradient_collateral = abs(float(gradient[0].mean()) - 0.5)
        self.assertLess(gradient_collateral, even_collateral * 0.5)

    def test_even_mode_is_unchanged_default(self):
        tiles, metadata = self._three_tile_setup()
        default = _cross_tile_overlap_consistency(tiles, metadata, 100)
        even = _cross_tile_overlap_consistency(tiles, metadata, 100, "even")
        self.assertTrue(torch.allclose(default, even, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
