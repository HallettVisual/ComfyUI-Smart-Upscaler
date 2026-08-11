"""Deterministic surface-region detection and context delivery.

These tests protect the core anti-hallucination promise: a flat water/sky tile
is never left to guess its own identity. The pixels prove where the large
uniform regions are; the whole-image pass is required to name each one; and
every tile inside a measured region receives the same canonical identity.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from nodes.cache import SmartCachedTextGenerate, _global_caption_problem
from nodes.prompting import (
    SmartTilePromptResolver,
    _strip_unsupported_atmosphere,
    _surface_prompt_problem,
)
from nodes.regions import (
    color_name,
    flat_regions,
    measured_regions_text,
    uncovered_region_problem,
    uniform_tile_components,
)
from nodes.universal_prompting import SmartUnifiedPromptGuidance, _build_all_tile_jobs


def _noisy_city_image_with_flat_blue_bottom_left(size=240):
    torch.manual_seed(7)
    image = torch.rand(size, size, 3)
    image[int(size * 0.62) :, : int(size * 0.46)] = torch.tensor([0.28, 0.42, 0.55])
    return image


class FlatRegionMeasurementTests(unittest.TestCase):
    def test_flat_blue_area_is_measured_and_located(self):
        regions = flat_regions(_noisy_city_image_with_flat_blue_bottom_left())
        self.assertEqual(len(regions), 1)
        self.assertIn("bottom left", regions[0]["labels"])
        self.assertIn("blue", regions[0]["color_name"])
        self.assertGreater(regions[0]["area_fraction"], 0.05)

    def test_fully_noisy_image_measures_no_flat_regions(self):
        torch.manual_seed(3)
        self.assertEqual(flat_regions(torch.rand(200, 200, 3)), [])

    def test_measured_text_demands_identification(self):
        regions = flat_regions(_noisy_city_image_with_flat_blue_bottom_left())
        text = measured_regions_text(regions)
        self.assertIn("MEASURED FLAT REGIONS", text)
        self.assertIn("bottom left", text)
        self.assertIn("MUST", text)

    def test_color_names_are_deterministic_measurements(self):
        self.assertEqual(color_name([0.5, 0.5, 0.5]), "gray")
        self.assertIn("blue", color_name([0.28, 0.42, 0.55]))
        self.assertEqual(color_name([0.05, 0.05, 0.06]), "near-black")

    def test_empty_surface_map_fails_coverage_and_matching_entry_passes(self):
        regions = flat_regions(_noisy_city_image_with_flat_blue_bottom_left())
        self.assertIn(
            "measured flat region", uncovered_region_problem([], regions)
        )
        covering = [{"locations": ["bottom-left", "bottom center"]}]
        self.assertEqual(uncovered_region_problem(covering, regions), "")


class UniformTileComponentTests(unittest.TestCase):
    def test_adjacent_same_color_uniform_tiles_form_one_surface_run(self):
        blue = [0.28, 0.42, 0.55]
        tiles = [
            {"tile_index": 0, "row": 0, "column": 0, "evidence_class": "structured", "mean_rgb": [0.5, 0.5, 0.5]},
            {"tile_index": 1, "row": 0, "column": 1, "evidence_class": "uniform", "mean_rgb": [0.9, 0.9, 0.9]},
            {"tile_index": 2, "row": 1, "column": 0, "evidence_class": "uniform", "mean_rgb": blue},
            {"tile_index": 3, "row": 1, "column": 1, "evidence_class": "uniform", "mean_rgb": blue},
            {"tile_index": 4, "row": 1, "column": 2, "evidence_class": "sparse", "mean_rgb": [0.30, 0.43, 0.56]},
        ]
        components = uniform_tile_components(tiles)
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]["members"], [2, 3])
        # The sparse neighbor (a boat edge on the same water) is attached, so it
        # can be offered the same surface identity as a candidate.
        self.assertEqual(components[0]["attached"], [4])

    def test_different_colored_uniform_tiles_do_not_merge(self):
        tiles = [
            {"tile_index": 0, "row": 0, "column": 0, "evidence_class": "uniform", "mean_rgb": [0.2, 0.4, 0.6]},
            {"tile_index": 1, "row": 0, "column": 1, "evidence_class": "uniform", "mean_rgb": [0.8, 0.8, 0.85]},
        ]
        self.assertEqual(uniform_tile_components(tiles), [])


class RegionIdentityDeliveryTests(unittest.TestCase):
    def _grid_jobs(self, global_context):
        """3x3 grid: noisy city everywhere except flat blue water at tiles 7 and 8."""
        torch.manual_seed(11)
        tiles = torch.rand(9, 64, 64, 3)
        water = torch.tensor([0.28, 0.42, 0.55])
        tiles[6] = water  # bottom left
        tiles[7] = water  # bottom center
        records = []
        for index in range(9):
            row, column = divmod(index, 3)
            records.append(
                {
                    "tile_index": index,
                    "source_index": 0,
                    "row": row,
                    "column": column,
                    "position": "center",
                    "x": column * 64,
                    "y": row * 64,
                    "width": 64,
                    "height": 64,
                }
            )
        metadata = json.dumps(
            {
                "image_width": 192,
                "image_height": 192,
                "tile_width": 64,
                "tile_height": 64,
                "tiles": records,
            }
        )
        _, prompt_system, _ = SmartUnifiedPromptGuidance().build()
        return _build_all_tile_jobs(
            tiles, metadata, json.dumps(global_context), prompt_system
        )

    def test_every_tile_of_a_measured_water_run_gets_the_surface_identity(self):
        # The vision model located the water at "bottom left" only. The measured
        # run covers bottom left AND bottom center, so both tiles must receive
        # the same canonical surface even though one location label was missed.
        water_prompt = (
            "calm open harbor water with a consistent deep blue-gray surface"
        )
        global_context = {
            "scene_type": "aerial urban waterfront",
            "geographic_context": "",
            "view": "high-angle aerial",
            "surface_map": [
                {
                    "id": "open_water",
                    "locations": ["bottom left"],
                    "identity": "open harbor water",
                    "target_prompt": water_prompt,
                }
            ],
            "object_map": [],
        }
        _, instructions, references = self._grid_jobs(global_context)

        for index in (6, 7):
            reference = json.loads(references[index])
            self.assertEqual(reference["evidence_class"], "uniform", index)
            surfaces = reference.get("canonical_surfaces", [])
            self.assertTrue(surfaces, f"tile {index} received no surface identity")
            self.assertEqual(surfaces[0]["id"], "open_water")
            self.assertEqual(surfaces[0]["target_prompt"], water_prompt)
            self.assertEqual(
                reference.get("canonical_surface_source"), "measured_uniform_region"
            )
            self.assertIn(water_prompt, instructions[index])

        # Whole-image location labels never enter tile instructions; the tile
        # model echoes them as in-tile positions ("open water at bottom center").
        for index in (6, 7):
            self.assertNotIn("matching_locations", instructions[index])

        # Structured city tiles are never stamped with the measured surface run.
        # (Nearby structured tiles may still see it as an unconfirmed candidate,
        # which the resolver ignores for structured content.)
        for index in (0, 1, 2, 3, 4, 5, 8):
            reference = json.loads(references[index])
            self.assertNotIn("canonical_surface_source", reference, index)
        # Tiles that do not touch the water's area receive no water candidate.
        for index in (0, 1, 2, 5, 8):
            reference = json.loads(references[index])
            for surface in reference.get("canonical_surfaces", []):
                self.assertNotEqual(surface.get("id"), "open_water", index)

    def test_empty_surface_map_still_leaves_uniform_tiles_scene_label_free(self):
        global_context = {
            "scene_type": "aerial urban waterfront",
            "geographic_context": "likely a coastal city",
            "view": "high-angle aerial",
            "surface_map": [], "material_map": [],
            "object_map": [],
        }
        _, instructions, references = self._grid_jobs(global_context)
        for index in (6, 7):
            self.assertEqual(
                json.loads(references[index])["evidence_class"], "uniform"
            )
            self.assertNotIn("waterfront", instructions[index])
            self.assertNotIn("coastal city", instructions[index])


class WholeImageSurfaceReliabilityTests(unittest.TestCase):
    def test_skipped_measured_region_is_recovered_with_one_focused_question(self):
        lazy_brief = json.dumps(
            {
                "scene_type": "aerial urban waterfront",
                "geographic_context": "",
                "view": "high-angle aerial",
                "surface_map": [], "material_map": [],
                "object_map": [],
            }
        )

        class SkipsSurfacesThenAnswersModel:
            def __init__(self):
                self.calls = 0
                self.prompts = []

            def tokenize(self, prompt, **kwargs):
                self.prompts.append(str(prompt))
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                return [self.calls]

            def decode(self, generated_ids):
                if self.calls <= 2:
                    return lazy_brief
                return "Open water."

        image = _noisy_city_image_with_flat_blue_bottom_left().unsqueeze(0)
        global_instruction, prompt_system, _ = SmartUnifiedPromptGuidance().build()
        model = SkipsSurfacesThenAnswersModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch.dict(os.environ, {"SMART_UPSCALER_CACHE_DIR": cache_directory}):
                text, status, _ = SmartCachedTextGenerate().generate(
                    model,
                    image,
                    global_instruction,
                    1024,
                    "Managed by Prompt Director (recommended)",
                    False,
                    True,
                    "refresh",
                    "measured-region-test",
                    "",
                    prompt_system,
                )

        # The model saw the pixel measurement in its instruction.
        self.assertIn("MEASURED FLAT REGIONS", model.prompts[0])
        # Two brief attempts, one focused region question, then the main-subject
        # ask plus its one strict re-ask (this model never gives area labels).
        self.assertEqual(model.calls, 5)
        payload = json.loads(text)
        self.assertEqual(len(payload["surface_map"]), 1)
        entry = payload["surface_map"][0]
        self.assertEqual(entry["identity"], "Open water")
        self.assertIn("bottom left", entry["locations"])
        self.assertEqual(
            _global_caption_problem(text, prompt_system, flat_regions(image)), ""
        )
        self.assertIn("RETRY WRITE", status)

    def test_brief_that_already_covers_the_measured_region_passes_first_try(self):
        good_brief = json.dumps(
            {
                "scene_type": "aerial urban waterfront",
                "geographic_context": "",
                "view": "high-angle aerial",
                "surface_map": [
                    {
                        "id": "open_water",
                        "locations": ["bottom left", "bottom center"],
                        "identity": "open harbor water",
                        "target_prompt": "calm deep blue-gray harbor water",
                    }
                ],
                "material_map": [],
                "object_map": [],
            }
        )

        class GoodBriefModel:
            def __init__(self):
                self.calls = 0

            def tokenize(self, prompt, **kwargs):
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                return [self.calls]

            def decode(self, generated_ids):
                return good_brief

        image = _noisy_city_image_with_flat_blue_bottom_left().unsqueeze(0)
        global_instruction, prompt_system, _ = SmartUnifiedPromptGuidance().build()
        model = GoodBriefModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch.dict(os.environ, {"SMART_UPSCALER_CACHE_DIR": cache_directory}):
                text, status, _ = SmartCachedTextGenerate().generate(
                    model,
                    image,
                    global_instruction,
                    1024,
                    "Managed by Prompt Director (recommended)",
                    False,
                    True,
                    "refresh",
                    "measured-region-good-test",
                    "",
                    prompt_system,
                )

        # One brief attempt plus the main-subject question (object_map was empty).
        self.assertEqual(model.calls, 2)
        self.assertEqual(json.loads(text)["surface_map"][0]["id"], "open_water")
        self.assertIn("WRITE", status)


class SceneLabelHonestyTests(unittest.TestCase):
    """A hallucinated scene label ("waterfront" over a park) must never reach tiles."""

    def _structured_instruction(self, global_context):
        torch.manual_seed(9)
        tiles = torch.rand(1, 64, 64, 3)
        metadata = json.dumps(
            {
                "image_width": 64,
                "image_height": 64,
                "tile_width": 64,
                "tile_height": 64,
                "tiles": [
                    {
                        "tile_index": 0,
                        "source_index": 0,
                        "row": 0,
                        "column": 0,
                        "position": "center",
                        "x": 0,
                        "y": 0,
                        "width": 64,
                        "height": 64,
                    }
                ],
            }
        )
        _, prompt_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Google Image Enhance"
        )
        _, instructions, _ = _build_all_tile_jobs(
            tiles, metadata, json.dumps(global_context), prompt_system
        )
        return instructions[0]

    def test_unlicensed_waterfront_label_is_stripped_from_tile_context(self):
        instruction = self._structured_instruction(
            {
                "scene_type": "aerial urban waterfront",
                "geographic_context": "",
                "view": "high-angle aerial",
                "surface_map": [], "material_map": [],
                "object_map": [],
            }
        )
        self.assertNotIn("waterfront", instruction)
        self.assertIn("aerial urban", instruction)

    def test_waterfront_label_is_kept_when_the_surface_map_contains_water(self):
        instruction = self._structured_instruction(
            {
                "scene_type": "aerial urban waterfront",
                "geographic_context": "",
                "view": "high-angle aerial",
                "surface_map": [
                    {
                        "id": "open_water",
                        "locations": ["bottom left"],
                        "identity": "open water",
                        "target_prompt": "calm deep blue open water",
                    }
                ],
                "object_map": [],
            }
        )
        self.assertIn("aerial urban waterfront", instruction)

    def test_master_instruction_contains_no_copyable_scene_examples(self):
        global_instruction, _, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Google Image Enhance"
        )
        self.assertNotIn("aerial urban waterfront", global_instruction)
        self.assertNotIn("alpine lake", global_instruction)
        self.assertIn("ONLY when open water pixels are clearly visible", global_instruction)


class MainSubjectTrackingTests(unittest.TestCase):
    """A clear main subject (the camel) must become a canonical object, so no
    tile ever hedges ("camel-like") or treats a hump as background scenery."""

    def test_master_instruction_demands_the_main_subject(self):
        global_instruction, _, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer"
        )
        self.assertIn("main subject", global_instruction)
        self.assertIn("MUST have an entry here", global_instruction)
        self.assertNotIn("Usually empty", global_instruction)
        self.assertIn('never hedge', global_instruction)

    def test_missing_main_subject_is_recovered_with_one_focused_question(self):
        brief_without_subject = json.dumps(
            {
                "scene_type": "desert",
                "geographic_context": "",
                "view": "eye-level",
                "surface_map": [], "material_map": [],
                "object_map": [],
            }
        )

        class SkipsSubjectThenAnswersModel:
            def __init__(self):
                self.calls = 0
                self.prompts = []

            def tokenize(self, prompt, **kwargs):
                self.prompts.append(str(prompt))
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                return [self.calls]

            def decode(self, generated_ids):
                if self.calls == 1:
                    return brief_without_subject
                return (
                    "a standing camel | top left, top right, bottom left, bottom right"
                )

        torch.manual_seed(15)
        image = torch.rand(1, 200, 200, 3)  # fully noisy: no measured regions
        global_instruction, prompt_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer"
        )
        model = SkipsSubjectThenAnswersModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch.dict(os.environ, {"SMART_UPSCALER_CACHE_DIR": cache_directory}):
                text, _, _ = SmartCachedTextGenerate().generate(
                    model,
                    image,
                    global_instruction,
                    1024,
                    "Managed by Prompt Director (recommended)",
                    False,
                    True,
                    "refresh",
                    "main-subject-test",
                    "",
                    prompt_system,
                )

        self.assertEqual(model.calls, 2)
        entry = json.loads(text)["object_map"][0]
        self.assertEqual(entry["identity"], "a standing camel")
        self.assertEqual(
            entry["locations"],
            ["top left", "top right", "bottom left", "bottom right"],
        )

    def test_subject_parts_reach_every_tile_with_the_right_hint(self):
        # The camel covers all four quarters; each tile must receive the subject
        # candidate WITH the part that sits in its own area — even if the stated
        # location list had missed an area, the hypothesis is still offered.
        from nodes.universal_prompting import _object_map_context

        global_context = {
            "object_map": [
                {
                    "id": "main_subject",
                    "locations": ["top left", "top right", "bottom left"],
                    "parts": {
                        "top left": "the camel's back and hump",
                        "top right": "the camel's head and neck",
                        "bottom left": "the camel's hind legs",
                    },
                    "identity": "a standing camel",
                    "target_prompt": "a single standing camel with tan fur",
                }
            ]
        }
        metadata = {"image_width": 100, "image_height": 100}

        def tile(x, y):
            return {"x": x, "y": y, "width": 50, "height": 50}

        top_left = _object_map_context(global_context, tile(0, 0), metadata)
        self.assertEqual(top_left[0]["part_in_this_tile"], "the camel's back and hump")
        top_right = _object_map_context(global_context, tile(50, 0), metadata)
        self.assertEqual(top_right[0]["part_in_this_tile"], "the camel's head and neck")
        # Bottom right was MISSING from locations: the candidate is still offered
        # (pixel confirmation remains the gate), just without a part hint.
        bottom_right = _object_map_context(global_context, tile(50, 50), metadata)
        self.assertEqual(bottom_right[0]["identity"], "a standing camel")
        self.assertNotIn("part_in_this_tile", bottom_right[0])
        # Whole-image location labels still never leak into candidates.
        self.assertNotIn("matching_locations", top_left[0])

    def test_a_declared_surface_is_never_offered_as_an_object_part(self):
        # Toronto waterfront read, 2026-07-30. The whole-image pass gave the
        # picture-spanning skyline every region, then filled the water regions
        # of its parts map with "water". That handed the bottom-left water tile
        # a part hint of "water", which it confirmed from its own pixels - and
        # the edit engine painted a glass building into the lake.
        from nodes.universal_prompting import _object_map_context

        global_context = {
            "surface_map": [
                {
                    "id": "water",
                    "locations": ["bottom left", "bottom center"],
                    "identity": "water",
                    "target_prompt": "water",
                }
            ],
            "object_map": [
                {
                    "id": "main subject",
                    "locations": [
                        "top left",
                        "top center",
                        "top right",
                        "middle left",
                        "middle center",
                        "middle right",
                        "bottom left",
                        "bottom center",
                        "bottom right",
                    ],
                    "parts": {
                        "top left": "high-rise buildings",
                        "middle left": "high-rise buildings",
                        "bottom left": "water",
                        "bottom center": "water",
                        "bottom right": "water",
                    },
                    "identity": "urban skyline with glass and concrete buildings",
                    "target_prompt": (
                        "glass and concrete buildings with reflective facades, sharp "
                        "edges, and uniform grid patterns of windows"
                    ),
                }
            ],
        }
        # A tile small enough to sit inside one region, as in the real 5x3 run.
        metadata = {"image_width": 300, "image_height": 300}

        def tile(x, y):
            return {"x": x, "y": y, "width": 100, "height": 100}

        # The water tile: no part hint, and the surface region contributes no
        # overlap, so the resolver's location route cannot fire either.
        water_tile = _object_map_context(global_context, tile(0, 200), metadata)
        self.assertNotIn("part_in_this_tile", water_tile[0])
        self.assertEqual(water_tile[0]["spatial_overlap"], 0)

        # A tile the brief really did give buildings is untouched.
        building_tile = _object_map_context(global_context, tile(0, 0), metadata)
        self.assertEqual(building_tile[0]["part_in_this_tile"], "high-rise buildings")
        self.assertGreater(building_tile[0]["spatial_overlap"], 0)

    def test_focused_subject_answer_with_parts_builds_a_segmented_entry(self):
        brief_without_subject = json.dumps(
            {
                "scene_type": "desert",
                "geographic_context": "",
                "view": "eye-level",
                "surface_map": [], "material_map": [],
                "object_map": [],
            }
        )

        class AnswersWithPartsModel:
            def __init__(self):
                self.calls = 0

            def tokenize(self, prompt, **kwargs):
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                return [self.calls]

            def decode(self, generated_ids):
                if self.calls == 1:
                    return brief_without_subject
                return (
                    "a standing camel | top left: its back and hump; "
                    "top right: its head and neck; bottom left: its hind legs; "
                    "bottom right: its front legs"
                )

        torch.manual_seed(21)
        image = torch.rand(1, 200, 200, 3)
        global_instruction, prompt_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer"
        )
        model = AnswersWithPartsModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch.dict(os.environ, {"SMART_UPSCALER_CACHE_DIR": cache_directory}):
                text, _, _ = SmartCachedTextGenerate().generate(
                    model,
                    image,
                    global_instruction,
                    1024,
                    "Managed by Prompt Director (recommended)",
                    False,
                    True,
                    "refresh",
                    "subject-parts-test",
                    "",
                    prompt_system,
                )
        entry = json.loads(text)["object_map"][0]
        self.assertEqual(entry["identity"], "a standing camel")
        self.assertEqual(len(entry["locations"]), 4)
        self.assertEqual(entry["parts"]["top left"], "its back and hump")
        self.assertEqual(entry["parts"]["bottom right"], "its front legs")

    def test_subject_answer_parser_survives_messy_model_replies(self):
        parse = SmartCachedTextGenerate._parse_subject_answer

        # Exact requested format.
        identity, locations, parts = parse(
            "an orc warrior | center: its face and tusks; middle left: spiked shoulder"
        )
        self.assertEqual(identity, "an orc warrior")
        self.assertEqual(locations, ["middle left", "center"])
        self.assertEqual(parts["center"], "its face and tusks")

        # Comma separators and prose, no pipe at all.
        identity, locations, parts = parse(
            "The main subject is an orc. In the center, its scarred face, "
            "bottom left has its spiked shoulder armor, bottom right the fur cloak."
        )
        self.assertIn("bottom left", locations)
        self.assertIn("spiked shoulder armor", parts["bottom left"])

        # "top center" must never be double-counted as "center".
        identity, locations, parts = parse(
            "a lighthouse | top center: the lantern room; bottom center: the base"
        )
        self.assertEqual(locations, ["top center", "bottom center"])
        self.assertNotIn("center", parts)

        # A refusal stays a refusal.
        self.assertEqual(parse("none"), ("", [], {}))

    def test_lazy_subject_entry_gets_enriched_with_parts(self):
        # The exact camel failure: the model DID list the subject, but lazily -
        # one location, no parts. The focused question must still fire and merge
        # the segmented coverage into the existing entry.
        lazy_brief = json.dumps(
            {
                "scene_type": "desert",
                "geographic_context": "",
                "view": "eye-level",
                "surface_map": [], "material_map": [],
                "object_map": [
                    {
                        "id": "camel",
                        "locations": ["center"],
                        "identity": "camel",
                        "target_prompt": "camel standing in desert",
                    }
                ],
            }
        )

        class LazyThenSegmentedModel:
            def __init__(self):
                self.calls = 0

            def tokenize(self, prompt, **kwargs):
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                return [self.calls]

            def decode(self, generated_ids):
                if self.calls == 1:
                    return lazy_brief
                return (
                    "a standing camel | top left: its back and hump; "
                    "top right: its head and neck; bottom left: its hind legs; "
                    "bottom right: its front legs"
                )

        torch.manual_seed(22)
        image = torch.rand(1, 200, 200, 3)
        global_instruction, prompt_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer"
        )
        model = LazyThenSegmentedModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch.dict(os.environ, {"SMART_UPSCALER_CACHE_DIR": cache_directory}):
                text, _, _ = SmartCachedTextGenerate().generate(
                    model,
                    image,
                    global_instruction,
                    1024,
                    "Managed by Prompt Director (recommended)",
                    False,
                    True,
                    "refresh",
                    "lazy-subject-test",
                    "",
                    prompt_system,
                )
        entry = json.loads(text)["object_map"][0]
        # Original identity and prompt kept; coverage and parts merged in.
        self.assertEqual(entry["identity"], "camel")
        self.assertEqual(entry["target_prompt"], "camel standing in desert")
        self.assertIn("top left", entry["locations"])
        self.assertIn("bottom right", entry["locations"])
        self.assertEqual(entry["parts"]["top left"], "its back and hump")

    def test_malformed_entry_is_repaired_instead_of_killing_the_run(self):
        # locations as a bare string and no target_prompt: previously a hard
        # "object_map contains an invalid entry" crash after retry.
        broken_brief = json.dumps(
            {
                "scene_type": "desert",
                "geographic_context": "",
                "view": "eye-level",
                "surface_map": [], "material_map": [],
                "object_map": [
                    {"identity": "camel", "locations": "center"}
                ],
            }
        )

        class BrokenEntryModel:
            def __init__(self):
                self.calls = 0

            def tokenize(self, prompt, **kwargs):
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                return [self.calls]

            def decode(self, generated_ids):
                if self.calls <= 2:
                    return broken_brief
                return "none"

        torch.manual_seed(23)
        image = torch.rand(1, 200, 200, 3)
        global_instruction, prompt_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer"
        )
        model = BrokenEntryModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch.dict(os.environ, {"SMART_UPSCALER_CACHE_DIR": cache_directory}):
                text, _, _ = SmartCachedTextGenerate().generate(
                    model,
                    image,
                    global_instruction,
                    1024,
                    "Managed by Prompt Director (recommended)",
                    False,
                    True,
                    "refresh",
                    "broken-entry-test",
                    "",
                    prompt_system,
                )
        entry = json.loads(text)["object_map"][0]
        self.assertEqual(entry["identity"], "camel")
        self.assertEqual(entry["locations"], ["center"])
        self.assertEqual(entry["target_prompt"], "camel")
        self.assertTrue(entry["id"])

    def test_answer_none_leaves_the_object_map_empty(self):
        empty_brief = json.dumps(
            {
                "scene_type": "open field",
                "geographic_context": "",
                "view": "aerial",
                "surface_map": [], "material_map": [],
                "object_map": [],
            }
        )

        class NoSubjectModel:
            def __init__(self):
                self.calls = 0

            def tokenize(self, prompt, **kwargs):
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                return [self.calls]

            def decode(self, generated_ids):
                return empty_brief if self.calls == 1 else "none"

        torch.manual_seed(16)
        image = torch.rand(1, 200, 200, 3)
        global_instruction, prompt_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer"
        )
        model = NoSubjectModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch.dict(os.environ, {"SMART_UPSCALER_CACHE_DIR": cache_directory}):
                text, _, _ = SmartCachedTextGenerate().generate(
                    model,
                    image,
                    global_instruction,
                    1024,
                    "Managed by Prompt Director (recommended)",
                    False,
                    True,
                    "refresh",
                    "no-subject-test",
                    "",
                    prompt_system,
                )
        self.assertEqual(json.loads(text)["object_map"], [])


class SamplerFamilySplitTests(unittest.TestCase):
    """Instruction models get an edit command; denoise models get a description."""

    def _resolve(self, prompt_system):
        return SmartTilePromptResolver().resolve(
            json.dumps(
                {
                    "local_caption": "glass towers beside a plaza",
                    "dominant_region": "glass towers",
                    "visible_boundaries": "",
                    "surface_id": "",
                    "surface_prompt": "",
                    "object_id": "",
                    "object_prompt": "",
                    "local_features": "",
                    "corrections_applied": "",
                    "target_prompt": (
                        "Tall blue glass towers on the left, a paved stone plaza with "
                        "planters on the right"
                    ),
                }
            ),
            json.dumps({"tile_index": 0, "tile_id": "T001", "evidence_class": "structured"}),
            prompt_system,
            "artifacts, seams",
        )

    def test_instruction_style_keeps_the_edit_command(self):
        _, system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
        )
        positive, _, _, _ = self._resolve(system)
        self.assertTrue(positive.startswith("Upscale this image"))

    def test_description_style_drops_the_edit_command_for_denoise_samplers(self):
        _, system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
            sampler_prompt_style="Plain description (SDXL, Flux, denoise)",
        )
        positive, negative, _, _ = self._resolve(system)
        self.assertNotIn("Upscale this image", positive)
        self.assertTrue(positive.startswith("Tall blue glass towers"))
        # Denoise samplers rely on the negative prompt; it must be populated.
        self.assertIn("new objects", negative)

    def test_maximum_detail_level_exists_with_its_own_budget(self):
        from nodes.universal_prompting import (
            TILE_DETAIL_CHOICES,
            _tile_detail_contract,
        )

        self.assertIn("Maximum (every visible item)", TILE_DETAIL_CHOICES)
        limit, rule = _tile_detail_contract(
            "Maximum (every visible item)", "structured", "dense"
        )
        self.assertEqual(limit, 100)
        self.assertIn("EVERY visible item", rule)

    def test_sdxl_preset_ships_with_maximum_and_description_style(self):
        entry = _installed_preset(
            self, "SDXL & Flux - Describe Everything (max detail)"
        )
        self.assertEqual(entry["tileDetail"], "Maximum (every visible item)")
        self.assertEqual(entry["samplerStyle"], "Plain description (SDXL, Flux, denoise)")
        self.assertIn("Name every", entry["instructions"])


def _installed_preset(test, name):
    """A tuned preset by name, skipping the test when no pack provides it.

    The shipped preset file is empty: every tuned instruction set is add-on
    content that lives outside this repo. These checks are still worth running
    on a machine that has a pack installed, so they skip rather than vanish.
    """
    import preset_store

    entry = preset_store.load_builtin_task_presets().get(name)
    if entry is None:
        test.skipTest(f"no installed preset pack provides {name!r}")
    return entry


class PromptSuffixTests(unittest.TestCase):
    """The Tile Prompt Suffix is plain prompt text attached to the end of every
    tile prompt - it rides after all guards, applies on cached and deterministic
    paths alike (the resolver runs live), and never invalidates caption caches."""

    def _resolve(self, prompt_system):
        return SmartTilePromptResolver().resolve(
            json.dumps(
                {
                    "local_caption": "glass towers beside a plaza",
                    "dominant_region": "glass towers",
                    "visible_boundaries": "",
                    "surface_id": "",
                    "surface_prompt": "",
                    "object_id": "",
                    "object_prompt": "",
                    "local_features": "",
                    "corrections_applied": "",
                    "target_prompt": "Tall blue glass towers beside a paved plaza",
                }
            ),
            json.dumps({"tile_index": 0, "tile_id": "T001", "evidence_class": "structured"}),
            prompt_system,
            "artifacts, seams",
        )

    def test_default_suffix_lands_at_the_very_end(self):
        _, system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
        )
        positive, _, _, _ = self._resolve(system)
        self.assertTrue(positive.endswith(", Fine detail."), positive)

    def test_custom_suffix_and_description_mode(self):
        _, system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
            sampler_prompt_style="Plain description (SDXL, Flux, denoise)",
            prompt_suffix="sharp focus, 8k",
        )
        positive, _, _, _ = self._resolve(system)
        self.assertTrue(positive.startswith("Tall blue glass towers"))
        self.assertTrue(positive.endswith(", sharp focus, 8k."), positive)

    def test_empty_suffix_appends_nothing(self):
        _, system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
            prompt_suffix="",
        )
        positive, _, _, _ = self._resolve(system)
        self.assertNotIn("fine detail", positive)
        self.assertFalse(positive.rstrip(".").endswith(","), positive)

    def test_suffix_reaches_the_no_analysis_direct_path(self):
        _, system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
            user_request="Upscale and add subtle detail to this image",
        )
        system = dict(system, prompt_strategy="direct_user")
        positive, _, _, _ = self._resolve(system)
        self.assertTrue(positive.endswith(", Fine detail."), positive)

    def test_legacy_prompt_systems_without_the_key_are_untouched(self):
        _, system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
        )
        system.pop("prompt_suffix", None)
        positive, _, _, _ = self._resolve(system)
        self.assertNotIn("Fine detail", positive)

    def test_old_widget_order_values_shift_back_into_place(self):
        # A graph saved before the suffix box moved above the dropdowns feeds
        # its dropdown values one slot late; build() must un-shift them.
        _, system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
            prompt_suffix="Maximum (every visible item)",
            tile_detail="Plain description (SDXL, Flux, denoise)",
        )
        self.assertEqual(system["caption_detail"], "Maximum (every visible item)")
        self.assertEqual(system["prompt_format"], "description")
        self.assertEqual(system["prompt_suffix"], "Fine detail")

    def test_preset_pack_families_cover_both_sampler_sections(self):
        import preset_store

        pack = preset_store.load_builtin_task_presets()
        if not pack:
            self.skipTest("no installed preset pack")
        families = {entry.get("family") for entry in pack.values()}
        self.assertEqual(families, {"edit", "denoise"})
        denoise = {
            name: entry
            for name, entry in pack.items()
            if entry.get("family") == "denoise"
        }
        self.assertIn("SDXL & Flux - Upscale / Detailer", denoise)
        self.assertIn("SDXL & Flux - Face & Portrait", denoise)
        self.assertIn("Z-Turbo - Generate Upscale", denoise)
        for entry in denoise.values():
            self.assertEqual(
                entry.get("samplerStyle"), "Plain description (SDXL, Flux, denoise)"
            )

    def test_every_pack_preset_sets_every_director_field(self):
        """A preset fills the whole node, so no field can linger from the last
        preset. The four base names are engine fallbacks and stay as they are."""
        import preset_store
        from nodes.universal_prompting import TILE_DETAIL_CHOICES

        pack = preset_store.load_builtin_task_presets()
        for name, entry in pack.items():
            with self.subTest(preset=name):
                self.assertIn("instructions", entry)
                self.assertIn(entry.get("family"), ("edit", "denoise"))
                from nodes.universal_prompting import TILE_COLOR_CHOICES

                self.assertIn(entry.get("tileColors"), TILE_COLOR_CHOICES)
                if "tileDetail" in entry:
                    self.assertIn(entry["tileDetail"], TILE_DETAIL_CHOICES)

    def test_z_turbo_preset_is_a_denoise_generate_upscale(self):
        entry = _installed_preset(self, "Z-Turbo - Generate Upscale")
        self.assertEqual(entry["family"], "denoise")
        self.assertEqual(entry["samplerStyle"], "Plain description (SDXL, Flux, denoise)")
        self.assertEqual(entry["tileDetail"], "Detailed")
        self.assertTrue(entry["instructions"].startswith("TASK: Upscale / Detailer"))
        # It asks for materials, and it keeps the project's blur stance.
        self.assertIn("material", entry["instructions"])
        self.assertIn("out-of-focus areas stay just as soft", entry["instructions"])

    def test_no_color_words_preset_carries_the_strip(self):
        from nodes.universal_prompting import TILE_COLOR_CHOICES

        entry = _installed_preset(
            self, "Upscale - No Color Words (fixes tile tinting)"
        )
        self.assertEqual(entry["tileColors"], TILE_COLOR_CHOICES[1])
        system = SmartUnifiedPromptGuidance().build(
            instructions=entry["instructions"],
            tile_colors=entry["tileColors"],
        )[1]
        self.assertEqual(system["color_words"], "strip")

    def test_color_repair_is_a_true_bypass_unless_switched_on(self):
        """Off must change nothing, and anything unrecognised must also mean
        off - a prompt-altering guard may never switch itself on."""
        from nodes.universal_prompting import TILE_COLOR_CHOICES, _color_words_mode

        for value in (
            TILE_COLOR_CHOICES[0],
            "",
            None,
            "Automatic (follow the preset instructions)",  # pre-v15.11 label
            "something nobody has written yet",
        ):
            with self.subTest(value=value):
                self.assertEqual(_color_words_mode(value), "keep")
        for value in (
            TILE_COLOR_CHOICES[1],
            "No color words (keep original colors)",  # pre-v15.11 label
        ):
            with self.subTest(value=value):
                self.assertEqual(_color_words_mode(value), "strip")


class SubjectAnswerJunkFilterTests(unittest.TestCase):
    """Chatty subject answers ('top center: etc. However, the image is...')
    must keep the locations but drop the junk part fragments."""

    def test_etc_and_trailing_prose_are_not_parts(self):
        parse = SmartCachedTextGenerate._parse_subject_answer
        identity, locations, parts = parse(
            "man | top center: etc. However, the image is a portrait, and the "
            "subject is the man's face and )"
        )
        self.assertEqual(identity, "man")
        self.assertIn("top center", locations)
        self.assertEqual(parts, {})

    def test_real_parts_still_parse(self):
        parse = SmartCachedTextGenerate._parse_subject_answer
        _, locations, parts = parse(
            "camel | top left: head and neck; top right: hump; "
            "bottom left: front legs; bottom right: hind legs"
        )
        self.assertEqual(parts["top left"], "head and neck")
        self.assertEqual(parts["bottom right"], "hind legs")
        self.assertEqual(len(locations), 4)


class UnlicensedSurfaceClaimTests(unittest.TestCase):
    """The fjord failure: blue mountain mist tiles guessed 'dark blue water'
    although the validated brief placed water at the bottom only. An ambiguous
    tile may claim a surface family (sky, water) ONLY when a location-matched
    canonical candidate delivered it; otherwise the guess is neutralized to the
    tile's literal soft colors. The user's own request words always win."""

    _FJORD_BRIEF = {
        "scene_type": "lake",
        "surface_map": [
            {
                "id": "open_sky",
                "locations": ["top left", "top center", "top right"],
                "identity": "open sky",
                "target_prompt": "muted teal open sky with even soft light",
            },
            {
                "id": "open_water",
                "locations": ["bottom left", "bottom center", "bottom right"],
                "identity": "open water",
                "target_prompt": "calm dark blue open water with soft ripples",
            },
        ],
        "material_map": [],
        "object_map": [],
    }

    @staticmethod
    def _metadata(y, height):
        # Vertical bands are top 0-40%, middle 40-60%, bottom 60-100% of the
        # 30px image. The CORE rect sits fully inside one band; the padded
        # rect pokes 1px past it, like real overlap at high scales.
        return {
            "image_width": 30,
            "image_height": 30,
            "tile_width": 30,
            "tile_height": height + 2,
            "output_tile_width": 30,
            "output_tile_height": height + 2,
            "tiles": [
                {
                    "tile_index": 0,
                    "source_index": 0,
                    "row": y // 10,
                    "column": 0,
                    "position": "middle left",
                    "x": 0,
                    "y": y - 1,
                    "width": 30,
                    "height": height + 2,
                    "core_x": 0,
                    "core_y": y,
                    "core_width": 30,
                    "core_height": height,
                }
            ],
        }

    def _reference(self, y, height=4):
        _, prompt_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
        )
        flat = torch.full((1, height + 2, 30, 3), 0.3)
        _, _, references = _build_all_tile_jobs(
            flat,
            json.dumps(self._metadata(y, height)),
            json.dumps(self._FJORD_BRIEF),
            prompt_system,
        )
        return json.loads(references[0])

    def test_middle_band_tile_has_no_license_for_sky_or_water(self):
        # Core 13..17 (43-57%) is pure middle band; the padded rect pokes
        # into top and bottom bands and must NOT attract their surfaces.
        reference = self._reference(y=13)
        self.assertEqual(reference["evidence_class"], "uniform")
        self.assertNotIn("canonical_surfaces", reference)
        self.assertEqual(
            reference["unlicensed_surface_families"], ["sky", "water"]
        )

    def test_bottom_tile_is_licensed_for_water_but_not_sky(self):
        reference = self._reference(y=22)
        self.assertEqual(reference["canonical_surfaces"][0]["id"], "open_water")
        self.assertEqual(reference["unlicensed_surface_families"], ["sky"])

    def _resolve(self, reference_extra, target, user_request=""):
        _, prompt_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
            user_request=user_request,
        )
        return SmartTilePromptResolver().resolve(
            json.dumps(
                {
                    "local_caption": target,
                    "dominant_region": target,
                    "visible_boundaries": "",
                    "surface_id": "",
                    "surface_prompt": "",
                    "object_id": "",
                    "object_prompt": "",
                    "local_features": "",
                    "corrections_applied": "",
                    "target_prompt": target,
                }
            ),
            json.dumps(
                {
                    "tile_index": 10,
                    "tile_id": "T011",
                    "evidence_class": "uniform",
                    **reference_extra,
                }
            ),
            prompt_system,
            "artifacts, seams",
        )

    def test_unlicensed_water_guess_becomes_literal_soft_tones(self):
        positive, _, audit, _ = self._resolve(
            {"unlicensed_surface_families": ["sky", "water"]},
            "dark blue water",
        )
        self.assertNotIn("water", positive.lower())
        self.assertIn(
            "dark blue tones in a smooth continuous gradient", positive.lower()
        )
        self.assertIn(
            "unlicensed surface-family claim neutralized",
            json.loads(audit)["corrections_applied"],
        )

    def test_unlicensed_sky_guess_is_neutralized_too(self):
        positive, _, _, _ = self._resolve(
            {"unlicensed_surface_families": ["sky", "water"]},
            "muted teal sky",
        )
        self.assertNotIn("sky", positive.lower())
        self.assertIn("muted teal tones", positive.lower())

    def test_user_request_words_always_keep_the_family(self):
        positive, _, _, _ = self._resolve(
            {"unlicensed_surface_families": ["sky", "water"]},
            "dark blue water",
            user_request="make the water look crisp",
        )
        self.assertIn("water", positive.lower())

    def test_reference_without_licensing_info_is_untouched(self):
        positive, _, _, _ = self._resolve({}, "dark blue water")
        self.assertIn("dark blue water", positive.lower())


class MaterialConsistencyTests(unittest.TestCase):
    """Textured continuous materials (fabric, brickwork, foliage) get ONE shared
    wording offered to textured tiles as candidates - the tile's own caption
    stays primary and the resolver never overrides it with a material."""

    _MATERIAL = {
        "id": "M1",
        "locations": ["bottom left", "bottom center", "bottom right"],
        "identity": "dark navy knit scarf fabric",
        "target_prompt": (
            "dark navy knit scarf fabric with a soft chunky ribbed weave of "
            "fine wool fibers, medium pattern scale"
        ),
    }

    @staticmethod
    def _metadata():
        return {
            "image_width": 32,
            "image_height": 24,
            "tile_width": 16,
            "tile_height": 8,
            "output_tile_width": 16,
            "output_tile_height": 8,
            "tiles": [
                {
                    "tile_index": 0,
                    "source_index": 0,
                    "row": 2,
                    "column": 0,
                    "position": "bottom left",
                    "x": 0,
                    "y": 16,
                    "width": 16,
                    "height": 8,
                    "core_x": 0,
                    "core_y": 16,
                    "core_width": 16,
                    "core_height": 8,
                }
            ],
        }

    def _jobs(self, tile_pixels, material):
        _, prompt_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
        )
        global_context = {"material_map": [material]}
        return _build_all_tile_jobs(
            tile_pixels,
            json.dumps(self._metadata()),
            json.dumps(global_context),
            prompt_system,
        )

    def test_whole_image_brief_demands_a_material_map(self):
        global_instruction, _, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
        )
        self.assertIn("material_map", global_instruction)
        self.assertIn("weave or pattern", global_instruction)
        self.assertIn(
            "surface_map, material_map, object_map", global_instruction
        )

    def test_material_map_is_a_hard_rule_not_an_invitation(self):
        # A brief that skips the question entirely is rejected (retry fires);
        # answering with an empty list is a legal answer.
        from nodes.cache import _global_caption_problem, _apply_global_brief_guard

        _, system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
        )
        skipped = json.dumps(
            {"scene_type": "desert", "surface_map": [], "object_map": []}
        )
        self.assertIn("material_map", _global_caption_problem(skipped, system, []))
        answered_empty = json.dumps(
            {
                "scene_type": "desert",
                "surface_map": [],
                "material_map": [],
                "object_map": [],
            }
        )
        self.assertEqual(_global_caption_problem(answered_empty, system, []), "")
        # The deterministic guard backfills the array on the recovery path.
        guarded = json.loads(_apply_global_brief_guard(skipped))
        self.assertEqual(guarded["material_map"], [])

    def test_unsafe_material_wording_sanitizes_to_its_identity(self):
        from nodes.cache import _sanitize_surface_map

        brief = json.dumps(
            {
                "surface_map": [],
                "material_map": [
                    {
                        "id": "M1",
                        "locations": ["bottom left"],
                        "identity": "red patterned rug",
                        "target_prompt": (
                            "red patterned rug area near surrounding furniture"
                        ),
                    }
                ],
                "object_map": [],
            }
        )
        cleaned = json.loads(_sanitize_surface_map(brief))
        self.assertEqual(
            cleaned["material_map"][0]["target_prompt"], "red patterned rug"
        )

    def test_textured_tile_receives_the_material_as_candidate_context(self):
        torch.manual_seed(7)
        textured = torch.rand((1, 8, 16, 3))
        _, instructions, references = self._jobs(textured, self._MATERIAL)
        reference = json.loads(references[0])
        self.assertEqual(reference["evidence_class"], "structured")
        self.assertEqual(reference["canonical_materials"][0]["id"], "M1")
        self.assertIn("canonical_material_candidates", instructions[0])
        self.assertIn("chunky ribbed weave", instructions[0])
        # Candidate wording, never an override: the instruction says so.
        self.assertIn("never an\noverride", instructions[0].replace("\r", ""))

    def test_uniform_tile_never_receives_material_candidates(self):
        flat = torch.full((1, 8, 16, 3), 0.25)
        _, instructions, references = self._jobs(flat, self._MATERIAL)
        reference = json.loads(references[0])
        self.assertEqual(reference["evidence_class"], "uniform")
        self.assertNotIn("canonical_materials", reference)
        self.assertNotIn("chunky ribbed weave", instructions[0])

    def test_unsafe_material_wording_is_dropped_not_delivered(self):
        torch.manual_seed(7)
        textured = torch.rand((1, 8, 16, 3))
        unsafe = dict(
            self._MATERIAL,
            target_prompt="dark fabric with reflections of surrounding buildings",
        )
        _, instructions, references = self._jobs(textured, unsafe)
        reference = json.loads(references[0])
        self.assertNotIn("canonical_materials", reference)
        self.assertNotIn("surrounding buildings", instructions[0])

    def test_resolver_keeps_the_tiles_own_caption_primary(self):
        # No regression: a structured tile with material candidates still
        # resolves from its OWN target_prompt; materials never override.
        _, prompt_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
        )
        positive, _, audit, _ = SmartTilePromptResolver().resolve(
            json.dumps(
                {
                    "local_caption": "dense knit scarf below a gray beard",
                    "dominant_region": "knit fabric",
                    "visible_boundaries": "",
                    "surface_id": "",
                    "surface_prompt": "",
                    "object_id": "",
                    "object_prompt": "",
                    "local_features": "",
                    "corrections_applied": "",
                    "target_prompt": (
                        "dark navy knit scarf fabric with a soft chunky ribbed "
                        "weave of fine wool fibers, gray beard strands above"
                    ),
                }
            ),
            json.dumps(
                {
                    "tile_index": 0,
                    "tile_id": "T001",
                    "evidence_class": "structured",
                    "canonical_materials": [self._MATERIAL],
                }
            ),
            prompt_system,
            "artifacts, seams",
        )
        self.assertIn("gray beard strands", positive)
        self.assertFalse(json.loads(audit).get("canonical_surface_override"))


class DeterministicUniformCaptionTests(unittest.TestCase):
    """A uniform tile with a stamped canonical surface never calls the VLM:
    the resolver would override its caption with the canonical text anyway."""

    def _reference(self, with_surface):
        reference = {
            "tile_index": 0,
            "tile_id": "T001",
            "evidence_class": "uniform",
        }
        if with_surface:
            reference["canonical_surfaces"] = [
                {
                    "id": "open_water",
                    "identity": "open water",
                    "source_appearance": "",
                    "target_prompt": "calm deep blue open water",
                }
            ]
        return json.dumps(reference)

    def test_uniform_canonical_tile_needs_no_model_at_all(self):
        from nodes.cache import SmartCachedTilePromptGenerator

        _, system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer"
        )
        node = SmartCachedTilePromptGenerator()
        image = torch.rand(1, 8, 8, 3)
        arguments = (
            image,
            "Inspect this exact tile.",
            self._reference(with_surface=True),
            system,
            "Managed by Prompt Director (recommended)",
            "read_write",
            "deterministic-uniform-test",
            "artifacts, seams",
        )
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch.dict(os.environ, {"SMART_UPSCALER_CACHE_DIR": cache_directory}):
                self.assertEqual(node.check_lazy_status(None, *arguments), [])
                result = node.generate(None, *arguments)
        self.assertIn("DETERMINISTIC", result[4])
        self.assertIn("calm deep blue open water", result[0])

    def test_uniform_tile_without_surface_still_needs_the_model(self):
        from nodes.cache import SmartCachedTilePromptGenerator

        _, system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer"
        )
        node = SmartCachedTilePromptGenerator()
        image = torch.rand(1, 8, 8, 3)
        arguments = (
            image,
            "Inspect this exact tile.",
            self._reference(with_surface=False),
            system,
            "Managed by Prompt Director (recommended)",
            "read_write",
            "deterministic-uniform-test-2",
            "artifacts, seams",
        )
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch.dict(os.environ, {"SMART_UPSCALER_CACHE_DIR": cache_directory}):
                self.assertEqual(node.check_lazy_status(None, *arguments), ["clip"])


class GhostSubjectRegressionTests(unittest.TestCase):
    """The old-man run: appending the whole-subject description to part tiles
    made a denoise sampler paint phantom faces in background tiles."""

    def _reference(self, with_hint):
        candidate = {
            "id": "main_subject",
            "identity": "elderly man with long gray hair and beard",
            "source_appearance": "",
            "target_prompt": (
                "Elderly man with long gray hair and beard, wearing a dark scarf "
                "and brown jacket"
            ),
        }
        if with_hint:
            candidate["part_in_this_tile"] = "his gray hair"
        return json.dumps(
            {
                "tile_index": 0,
                "tile_id": "T001",
                "evidence_class": "structured",
                "canonical_objects": [candidate],
            }
        )

    def _payload(self, target):
        return json.dumps(
            {
                "local_caption": "soft out-of-focus pale gradient with faint hair wisps",
                "dominant_region": "pale blurred gradient",
                "visible_boundaries": "",
                "surface_id": "",
                "surface_prompt": "",
                "object_id": "main_subject",
                "object_prompt": "",
                "local_features": "",
                "corrections_applied": "",
                "target_prompt": target,
            }
        )

    def test_description_mode_never_appends_the_whole_subject(self):
        _, system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
            sampler_prompt_style="Plain description (SDXL, Flux, denoise)",
        )
        positive, _, audit, _ = SmartTilePromptResolver().resolve(
            self._payload("Soft out-of-focus pale gradient with faint hair wisps"),
            self._reference(with_hint=True),
            system,
            "artifacts",
        )
        self.assertNotIn("wearing a dark scarf", positive)
        self.assertNotIn("Elderly man", positive)
        # The confirmed object is still recorded for the audit trail.
        self.assertEqual(json.loads(audit)["canonical_object_id"], "main_subject")

    def test_instruction_mode_appends_this_tile_s_own_part(self):
        """Instruction mode still appends for continuity - but the PART, not the
        whole subject.

        Appending the whole man to a tile showing only his hair is the failure
        this class was written for, and the edit path turned out not to be immune
        either: a brief bundled its subject as "dense cluster of tall buildings
        ... and a small park", and the waterfront park tile - whose part hint
        correctly read "waterfront park with trees and pathways" - was handed the
        tall buildings. Klein built them on the lawn.
        """
        _, system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
        )
        positive, _, _, _ = SmartTilePromptResolver().resolve(
            self._payload("The camel-free local content"),
            self._reference(with_hint=True),
            system,
            "artifacts",
        )
        self.assertIn("his gray hair", positive)
        self.assertNotIn("brown jacket", positive)

    def test_the_whole_subject_is_still_used_when_no_part_is_known(self):
        """Without a part hint there is nothing more specific to reach for, so
        the shared subject wording remains the continuity anchor."""
        _, system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
        )
        positive, _, _, _ = SmartTilePromptResolver().resolve(
            self._payload("The camel-free local content"),
            self._reference(with_hint=False),
            system,
            "artifacts",
        )
        self.assertIn("Elderly man with long gray hair", positive)

    def test_one_generic_token_cannot_confirm_without_an_expected_part(self):
        from nodes.prompting import _select_canonical_object

        candidate = {
            "id": "main_subject",
            "identity": "elderly man with long gray hair and beard",
            "target_prompt": "elderly man portrait",
        }
        # Only "hair" matches, and no part is expected here: rejected.
        rejected = _select_canonical_object(
            {"canonical_objects": [candidate]},
            {"object_prompt": "", "local_caption": "blurry hair wisps",
             "local_features": "", "dominant_region": "background"},
        )
        self.assertIsNone(rejected)
        # The same single token confirms when the subject's part is expected here.
        expected = dict(candidate, part_in_this_tile="his gray hair")
        confirmed = _select_canonical_object(
            {"canonical_objects": [expected]},
            {"object_prompt": "", "local_caption": "blurry hair wisps",
             "local_features": "", "dominant_region": "background"},
        )
        self.assertIsNotNone(confirmed)


class AnalysisImageCapTests(unittest.TestCase):
    def test_downscale_caps_long_side_and_keeps_aspect(self):
        from nodes.cache import _analysis_image

        image = torch.rand(1, 1500, 2000, 3)
        capped = _analysis_image(image, 1344)
        self.assertEqual(capped.shape[2], 1344)
        self.assertEqual(capped.shape[1], round(1500 * 1344 / 2000))
        # 0 disables the cap; small images pass through untouched.
        self.assertTrue(torch.equal(_analysis_image(image, 0), image))
        small = torch.rand(1, 500, 600, 3)
        self.assertTrue(torch.equal(_analysis_image(small, 1344), small))

    def test_vision_model_receives_the_capped_image(self):
        seen_shapes = []

        class ShapeRecordingModel:
            def __init__(self):
                self.calls = 0

            def tokenize(self, prompt, image=None, **kwargs):
                seen_shapes.append(tuple(image.shape))
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                return [self.calls]

            def decode(self, generated_ids):
                if self.calls == 1:
                    return json.dumps(
                        {
                            "scene_type": "field",
                            "geographic_context": "",
                            "view": "aerial",
                            "surface_map": [], "material_map": [],
                            "object_map": [],
                        }
                    )
                return "none"

        torch.manual_seed(31)
        image = torch.rand(1, 1500, 2000, 3)
        global_instruction, prompt_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer"
        )
        model = ShapeRecordingModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch.dict(os.environ, {"SMART_UPSCALER_CACHE_DIR": cache_directory}):
                SmartCachedTextGenerate().generate(
                    model,
                    image,
                    global_instruction,
                    1024,
                    "Managed by Prompt Director (recommended)",
                    False,
                    True,
                    "refresh",
                    "analysis-cap-test",
                    "",
                    prompt_system,
                )
        # Every question (brief + follow-ups) used the capped image.
        for shape in seen_shapes:
            self.assertEqual(max(shape[1], shape[2]), 1344, shape)


class SurfacePromptSafetyTests(unittest.TestCase):
    """A canonical surface phrase is stamped onto isolated tiles, so it must name a
    real material and never mention nearby objects — or the sampler paints them."""

    def test_object_mentions_and_process_words_are_rejected(self):
        bad_reflections = {
            "id": "s1",
            "locations": ["bottom left"],
            "identity": "open water",
            "target_prompt": (
                "smooth blue surface with subtle ripples and reflections of "
                "surrounding buildings"
            ),
        }
        self.assertIn("surrounding", _surface_prompt_problem(bad_reflections))

        echoed_measurement = {
            "id": "s1",
            "locations": ["bottom left"],
            "identity": "open water",
            "target_prompt": "smooth blue region with a calm surface",
        }
        self.assertIn("region", _surface_prompt_problem(echoed_measurement))

        good = {
            "id": "s1",
            "locations": ["bottom left"],
            "identity": "open water",
            "target_prompt": (
                "calm deep blue-gray open water with gentle natural ripples"
            ),
        }
        self.assertEqual(_surface_prompt_problem(good), "")

        # "reflecting sky" on a water surface is a flourish about something else.
        sky_flourish = {
            "id": "s1",
            "locations": ["bottom left"],
            "identity": "open water",
            "target_prompt": "open water, deep blue, calm, reflecting sky",
        }
        self.assertIn("sky", _surface_prompt_problem(sky_flourish))
        # A real sky surface may of course say sky.
        sky_surface = {
            "id": "s2",
            "locations": ["top left"],
            "identity": "overcast sky",
            "target_prompt": "soft pale gray overcast sky",
        }
        self.assertEqual(_surface_prompt_problem(sky_surface), "")

    def test_time_of_day_transform_keeps_requested_sky_language(self):
        # For a style/time-of-day transform, lighting and sky words in a surface
        # target are the requested result, not an invented flourish.
        night_water_brief = json.dumps(
            {
                "scene_type": "waterfront city",
                "geographic_context": "",
                "view": "high-angle aerial",
                "surface_map": [
                    {
                        "id": "open_water",
                        "locations": ["bottom left"],
                        "identity": "open water",
                        "target_prompt": (
                            "dark calm open water reflecting the night sky"
                        ),
                    }
                ],
                "material_map": [],
                "object_map": [],
            }
        )
        _, night_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Time of Day", user_request="Convert to night."
        )
        self.assertEqual(
            _global_caption_problem(night_water_brief, night_system, []), ""
        )
        # The same phrase in a faithful Google repair is rejected.
        _, google_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Google Image Enhance"
        )
        self.assertIn(
            "sky", _global_caption_problem(night_water_brief, google_system, [])
        )

    def test_repeated_unsafe_surface_prompt_is_sanitized_to_its_identity(self):
        bad_brief = json.dumps(
            {
                "scene_type": "aerial urban waterfront",
                "geographic_context": "",
                "view": "high-angle aerial",
                "surface_map": [
                    {
                        "id": "water",
                        "locations": ["bottom left"],
                        "identity": "open water",
                        "target_prompt": (
                            "smooth blue region with reflections of surrounding buildings"
                        ),
                    }
                ],
                "object_map": [],
            }
        )

        class RepeatsBadSurfaceModel:
            def __init__(self):
                self.calls = 0

            def tokenize(self, prompt, **kwargs):
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                return [self.calls]

            def decode(self, generated_ids):
                return bad_brief

        image = _noisy_city_image_with_flat_blue_bottom_left().unsqueeze(0)
        global_instruction, prompt_system, _ = SmartUnifiedPromptGuidance().build()
        model = RepeatsBadSurfaceModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch.dict(os.environ, {"SMART_UPSCALER_CACHE_DIR": cache_directory}):
                text, _, _ = SmartCachedTextGenerate().generate(
                    model,
                    image,
                    global_instruction,
                    1024,
                    "Managed by Prompt Director (recommended)",
                    False,
                    True,
                    "refresh",
                    "surface-sanitize-test",
                    "",
                    prompt_system,
                )

        entry = json.loads(text)["surface_map"][0]
        self.assertEqual(entry["target_prompt"], "open water")
        self.assertNotIn("buildings", json.dumps(entry))


class TilePromptHonestyTests(unittest.TestCase):
    def _google_system(self):
        _, system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Google Image Enhance"
        )
        return system

    def test_unsupported_sky_flourish_is_removed(self):
        self.assertNotIn(
            "sky",
            _strip_unsupported_atmosphere(
                "Urban waterfront with modern glass buildings, a dark waterway, "
                "and a partially visible dock with boats, all under a clear sky",
                "urban waterfront with buildings, waterway, dock, boats",
            ),
        )
        # Sky stays when the tile's own caption shows it.
        self.assertIn(
            "sky",
            _strip_unsupported_atmosphere(
                "street with buildings under an overcast sky",
                "buildings and overcast sky above the street",
            ),
        )

    def test_resolver_drops_sky_flourish_the_tile_does_not_show(self):
        positive, _, _, _ = SmartTilePromptResolver().resolve(
            json.dumps(
                {
                    "local_caption": "urban waterfront with buildings, a waterway, and a dock",
                    "dominant_region": "urban waterfront",
                    "visible_boundaries": "",
                    "surface_id": "",
                    "surface_prompt": "",
                    "object_id": "",
                    "object_prompt": "",
                    "local_features": "",
                    "corrections_applied": "",
                    "target_prompt": (
                        "Urban waterfront with modern glass buildings and a dark "
                        "waterway, all under a clear sky"
                    ),
                }
            ),
            json.dumps({"tile_index": 7, "tile_id": "T008", "evidence_class": "structured"}),
            self._google_system(),
            "artifacts",
        )
        self.assertNotIn("sky", positive.lower())
        self.assertIn("waterway", positive.lower())

    def test_google_repair_drops_soft_wear_language(self):
        positive, _, _, _ = SmartTilePromptResolver().resolve(
            json.dumps(
                {
                    "local_caption": "glass and concrete skyscrapers beside a highway",
                    "dominant_region": "skyscrapers",
                    "visible_boundaries": "",
                    "surface_id": "",
                    "surface_prompt": "",
                    "object_id": "",
                    "object_prompt": "",
                    "local_features": "",
                    "corrections_applied": "",
                    "target_prompt": (
                        "Glass and concrete skyscrapers, some with minor structural "
                        "wear, and a highway below"
                    ),
                }
            ),
            json.dumps({"tile_index": 3, "tile_id": "T004", "evidence_class": "structured"}),
            self._google_system(),
            "artifacts",
        )
        self.assertNotIn("wear", positive.lower())
        self.assertIn("skyscrapers", positive.lower())
        self.assertIn("highway", positive.lower())

    def test_unified_mode_delivers_the_preset_repair_rule_to_tiles(self):
        torch.manual_seed(2)
        tiles = torch.rand(1, 64, 64, 3)
        metadata = json.dumps(
            {
                "image_width": 64,
                "image_height": 64,
                "tile_width": 64,
                "tile_height": 64,
                "tiles": [
                    {
                        "tile_index": 0,
                        "source_index": 0,
                        "row": 0,
                        "column": 0,
                        "position": "center",
                        "x": 0,
                        "y": 0,
                        "width": 64,
                        "height": 64,
                    }
                ],
            }
        )
        _, prompt_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Google Image Enhance"
        )
        _, instructions, _ = _build_all_tile_jobs(
            tiles, metadata, "{}", prompt_system
        )
        self.assertIn("what SHOULD be there", instructions[0])
        self.assertIn("Never mention the\nsky, weather, sun, or atmosphere", instructions[0])


class UserDialAndPresetPackTests(unittest.TestCase):
    def _one_tile_jobs(self, prompt_system):
        torch.manual_seed(6)
        tiles = torch.rand(1, 64, 64, 3)
        metadata = json.dumps(
            {
                "image_width": 64,
                "image_height": 64,
                "tile_width": 64,
                "tile_height": 64,
                "tiles": [
                    {
                        "tile_index": 0,
                        "source_index": 0,
                        "row": 0,
                        "column": 0,
                        "position": "center",
                        "x": 0,
                        "y": 0,
                        "width": 64,
                        "height": 64,
                    }
                ],
            }
        )
        return _build_all_tile_jobs(tiles, metadata, "{}", prompt_system)

    def test_detail_level_is_the_only_dial_and_lives_outside_instructions(self):
        declared = SmartUnifiedPromptGuidance.INPUT_TYPES()
        self.assertEqual(
            list(declared["required"]),
            ["instructions", "user_request", "known_false_detections"],
        )
        # New controls append at the end so saved optional widget values never
        # shift into a different field.
        self.assertEqual(
            list(declared["optional"]),
            [
                "prompt_suffix",
                "tile_detail",
                "sampler_prompt_style",
                "tile_colors",
            ],
        )
        # A workflow saved before the control existed calls build without it.
        _, prompt_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
            user_request="",
            known_false_detections="",
        )
        self.assertEqual(
            prompt_system["caption_detail"], "Adaptive by Tile (recommended)"
        )
        # The dropdown, not any text line, decides the prompt complexity.
        _, prompt_system, _ = SmartUnifiedPromptGuidance().build(
            instructions="TASK: Upscale / Detailer",
            tile_detail="Complex",
        )
        self.assertEqual(prompt_system["caption_detail"], "Complex")

    def test_focused_upscale_presets_are_consistent_when_a_pack_is_installed(self):
        """Guards the tuned presets - which are an add-on, not part of the repo.

        The shipped preset file is empty by design, so on a clean clone there is
        nothing here to check and the test skips. On a machine with a pack
        installed it still catches a preset drifting away from its base task.
        """
        import preset_store
        from nodes.universal_prompting import _unified_task_name

        pack = preset_store.load_builtin_task_presets()
        for name, expected_blur_phrase in (
            ("Upscale - Sharpen Everything", "softness is a flaw"),
            ("Upscale - Face & Portrait", "stays just as soft"),
            ("Upscale - General with Color Prompting", "visible color of every major"),
        ):
            if name not in pack:
                continue
            body = pack[name]["instructions"]
            self.assertIn(expected_blur_phrase, body, name)
            # Every focused preset maps onto a base engine task via its TASK line.
            self.assertEqual(_unified_task_name(body), "Upscale / Detailer", name)
        # The sharpen preset must not contain the preserve-blur stance.
        if "Upscale - Sharpen Everything" in pack:
            self.assertNotIn(
                "stay just as soft", pack["Upscale - Sharpen Everything"]["instructions"]
            )

    def test_upscale_preset_teaches_blur_preservation(self):
        from nodes.universal_prompting import UNIFIED_TASK_INSTRUCTIONS

        body = UNIFIED_TASK_INSTRUCTIONS["Upscale / Detailer"]
        self.assertIn("out-of-focus areas stay just as soft", body)
        self.assertIn("never", body.lower())

    def test_the_four_base_tasks_live_in_code_not_in_the_preset_file(self):
        """The node must work with no presets installed at all.

        The shipped preset file is empty: presets are an add-on pack or the
        user's own saved entries. So every base task a TASK line can name has to
        resolve from code, or a fresh install breaks the moment someone runs it.
        """
        from nodes.universal_prompting import UNIFIED_TASK_INSTRUCTIONS

        self.assertEqual(
            set(UNIFIED_TASK_INSTRUCTIONS),
            {
                "Google Image Enhance",
                "Style Transfer",
                "Time of Day",
                "Upscale / Detailer",
            },
        )
        for name, body in UNIFIED_TASK_INSTRUCTIONS.items():
            self.assertTrue(body.strip(), name)

    def test_an_installed_pack_never_silently_rewords_a_base_task(self):
        """A pack may replace a base name - but then free and paid must agree.

        Packs load on top of the shipped file, so an entry reusing a base name
        overrides the code default. Identical wording that differs only in line
        breaks would give paid users a subtly different instruction than free
        users, for no reason and with nothing to show it had happened.
        """
        import preset_store
        from nodes.universal_prompting import UNIFIED_TASK_INSTRUCTIONS

        pack = preset_store.load_builtin_task_presets()
        for name in UNIFIED_TASK_INSTRUCTIONS:
            if name not in pack:
                continue
            self.assertEqual(
                pack[name]["instructions"].strip(),
                UNIFIED_TASK_INSTRUCTIONS[name].strip(),
                name,
            )

    def test_shipped_preset_file_is_empty_so_the_repo_carries_no_tuned_content(self):
        """The public repo ships code, not tuned instructions."""
        import json
        from pathlib import Path

        shipped = json.loads(
            (Path(__file__).resolve().parents[1] / "presets" / "task_presets.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(shipped["presets"], {})


if __name__ == "__main__":
    unittest.main()
