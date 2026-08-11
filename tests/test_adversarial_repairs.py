"""Regression tests for failure modes found by the adversarial audit."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from nodes.audit import _tile_health_lines, _tile_needed_recovery
from nodes.cache import (
    SmartCachedTilePromptGenerator,
    SmartCachedTextGenerate,
    _tile_context_fallback,
    _expected_object_problem,
    _global_caption_problem,
    _model_fingerprint,
    _repair_map_entries,
    _tensor_fingerprint,
    _write_prompt,
)
from nodes.prompting import (
    SmartTilePromptResolver,
    _excluded_detection_problem,
    _false_detection_forms,
    _map_prompt_is_thin,
    _select_canonical_object,
    _select_canonical_surface,
    _strip_color_words,
    _strip_frame_talk,
    _strip_reflected_objects,
    _strip_unsupported_atmosphere,
    _surface_covers_tile,
)
from nodes.regions import flat_regions, uniform_tile_components
from nodes.universal_prompting import (
    SmartTileJobDirector,
    SmartUnifiedPromptGuidance,
    _build_all_tile_jobs,
    _entry_contains_excluded,
    _local_pixel_evidence,
    _material_map_context,
    _parse_false_detections,
    _remove_false_detections,
    _object_map_context,
    _surface_map_context,
)


class ColorAwareEvidenceTests(unittest.TestCase):
    def test_isoluminant_color_boundary_is_not_uniform(self):
        image = torch.zeros(1, 96, 96, 3)
        # These channel means are identical, so the former grayscale average
        # erased the boundary completely.
        image[:, :, :48, 0] = 1.0
        image[:, :, 48:, 1] = 1.0
        evidence_class, _complexity, _metrics = _local_pixel_evidence(image)
        self.assertNotEqual(evidence_class, "uniform")

    def test_chromatic_checkerboard_is_not_a_flat_region(self):
        image = torch.zeros(1, 240, 240, 3)
        yy, xx = torch.meshgrid(torch.arange(240), torch.arange(240), indexing="ij")
        red = ((xx // 8 + yy // 8) % 2 == 0)
        image[0, red, 0] = 1.0
        image[0, ~red, 1] = 1.0
        self.assertEqual(flat_regions(image), [])


class SceneIsolationTests(unittest.TestCase):
    def test_uniform_components_never_cross_source_images(self):
        tiles = [
            {
                "tile_index": 0,
                "source_index": 0,
                "row": 0,
                "column": 0,
                "evidence_class": "uniform",
                "mean_rgb": [0.2, 0.4, 0.6],
            },
            {
                "tile_index": 1,
                "source_index": 1,
                "row": 0,
                "column": 1,
                "evidence_class": "uniform",
                "mean_rgb": [0.2, 0.4, 0.6],
            },
        ]
        self.assertEqual(uniform_tile_components(tiles), [])

    def test_job_builder_rejects_multi_source_metadata(self):
        metadata = {
            "image_batch": 2,
            "image_width": 16,
            "image_height": 16,
            "tile_width": 16,
            "tile_height": 16,
            "output_tile_width": 16,
            "output_tile_height": 16,
            "tiles": [
                {
                    "tile_index": index,
                    "source_index": index,
                    "row": 0,
                    "column": 0,
                    "position": "center",
                    "x": 0,
                    "y": 0,
                    "width": 16,
                    "height": 16,
                }
                for index in range(2)
            ],
        }
        with self.assertRaisesRegex(ValueError, "one source image at a time"):
            _build_all_tile_jobs(
                torch.zeros(2, 16, 16, 3),
                json.dumps(metadata),
                "{}",
                {"prompt_strategy": "direct_user", "direct_prompt": "enhance"},
            )


class SpatialSurfaceTests(unittest.TestCase):
    @staticmethod
    def _entry(entry_id, location):
        return {
            "id": entry_id,
            "locations": [location],
            "identity": entry_id.replace("_", " "),
            "source_appearance": "smooth",
            # A real shared description, not just the name repeated: a canonical
            # entry that only echoes its own identity is treated as absent.
            "target_prompt": f"{entry_id.replace('_', ' ')}, smooth and evenly lit",
        }

    def test_thin_canonical_surface_is_treated_as_absent(self):
        thin = {
            "id": "water",
            "locations": ["bottom center"],
            "identity": "water",
            "target_prompt": "water",
            "spatial_overlap": 0.9,
        }
        self.assertIsNone(
            _select_canonical_surface({"canonical_surfaces": [thin]}, {}, "uniform")
        )
        described = dict(
            thin, target_prompt="calm, deep teal water with gentle ripples"
        )
        chosen = _select_canonical_surface(
            {"canonical_surfaces": [described]}, {}, "uniform"
        )
        self.assertEqual(chosen["id"], "water")

    def test_thin_surface_is_ignored_even_when_the_tile_names_it(self):
        thin = {
            "id": "water",
            "locations": ["bottom center"],
            "identity": "water",
            "target_prompt": "water",
        }
        self.assertIsNone(
            _select_canonical_surface(
                {"canonical_surfaces": [thin]}, {"surface_id": "water"}, "uniform"
            )
        )

    def test_surface_ranking_uses_measured_overlap(self):
        context = {
            "surface_map": [
                self._entry("left_surface", "middle left"),
                self._entry("center_surface", "center"),
            ]
        }
        metadata = {"image_width": 100, "image_height": 100}
        tile = {"x": 0, "y": 40, "width": 70, "height": 20}
        candidates = _surface_map_context(context, tile, metadata)
        self.assertEqual(candidates[0]["id"], "left_surface")
        chosen = _select_canonical_surface(
            {"canonical_surfaces": candidates}, {}, "uniform"
        )
        self.assertEqual(chosen["id"], "left_surface")

    def test_equal_overlap_refuses_arbitrary_surface(self):
        context = {
            "surface_map": [
                self._entry("left_surface", "middle left"),
                self._entry("right_surface", "middle right"),
            ]
        }
        metadata = {"image_width": 100, "image_height": 100}
        tile = {"x": 0, "y": 40, "width": 100, "height": 20}
        candidates = _surface_map_context(context, tile, metadata)
        self.assertIsNone(
            _select_canonical_surface(
                {"canonical_surfaces": candidates}, {}, "uniform"
            )
        )


class PromptRecoveryTests(unittest.TestCase):
    @staticmethod
    def _whole_brief(object_map):
        return {
            "scene_type": "lake",
            "geographic_context": "",
            "view": "eye-level",
            "surface_map": [],
            "material_map": [],
            "object_map": object_map,
        }

    @staticmethod
    def _house_entry(parts=None):
        entry = {
            "id": "house",
            "locations": ["middle left", "center", "middle right"],
            "identity": "wooden cabin",
            "target_prompt": "wooden cabin with pitched roof and windows",
        }
        if parts is not None:
            entry["parts"] = parts
        return entry

    def test_global_brief_rejects_list_valued_object_parts(self):
        payload = self._whole_brief(
            [self._house_entry(["roof", "walls", "windows"])]
        )
        problem = _global_caption_problem(
            json.dumps(payload), {"unified_instruction_ui": True}
        )
        self.assertIn("object_map", problem)
        self.assertIn("invalid parts", problem)

    def test_global_brief_rejects_parts_outside_declared_locations(self):
        payload = self._whole_brief(
            [
                self._house_entry(
                    {
                        "top left": "roof",
                        "top center": "windows",
                        "top right": "balcony",
                    }
                )
            ]
        )
        problem = _global_caption_problem(
            json.dumps(payload), {"unified_instruction_ui": True}
        )
        self.assertIn("invalid parts", problem)

        repaired = json.loads(_repair_map_entries(json.dumps(payload)))
        self.assertNotIn("parts", repaired["object_map"][0])

    def test_global_brief_rejects_subject_template_placeholder(self):
        payload = self._whole_brief(
            [
                {
                    "id": "main_subject",
                    "locations": ["center"],
                    "identity": "<subject>",
                    "target_prompt": "<subject>",
                }
            ]
        )
        problem = _global_caption_problem(
            json.dumps(payload), {"unified_instruction_ui": True}
        )
        self.assertIn("template placeholder", problem)

    def test_map_repair_drops_invalid_parts_and_placeholder_entry(self):
        payload = self._whole_brief(
            [
                self._house_entry(["roof", "walls", "windows"]),
                {
                    "id": "main_subject",
                    "locations": ["center"],
                    "identity": "<subject>",
                    "target_prompt": "<subject>",
                },
            ]
        )
        repaired = json.loads(_repair_map_entries(json.dumps(payload)))
        self.assertEqual(len(repaired["object_map"]), 1)
        self.assertEqual(repaired["object_map"][0]["identity"], "wooden cabin")
        self.assertNotIn("parts", repaired["object_map"][0])

    def test_main_subject_recovery_ignores_echoed_placeholder(self):
        original = self._whole_brief([self._house_entry()])
        result = SmartCachedTextGenerate()._identify_main_subject(
            json.dumps(original),
            {"unified_instruction_ui": True},
            lambda _question, response_limit=None: (
                "<subject> | middle left: <part>; center: <part>"
            ),
        )
        self.assertEqual(json.loads(result), original)

    def test_bottom_left_cabin_gets_strong_object_overlap(self):
        metadata = {"image_width": 1024, "image_height": 1024}
        tile = {
            "core_x": 0,
            "core_y": 512,
            "core_width": 512,
            "core_height": 512,
            "x": 0,
            "y": 504,
            "width": 520,
            "height": 520,
        }
        context = self._whole_brief([self._house_entry()])
        candidates = _object_map_context(context, tile, metadata)
        self.assertEqual(candidates[0]["id"], "house")
        self.assertGreaterEqual(candidates[0]["spatial_overlap"], 0.19)

    def test_structured_caption_cannot_silently_omit_expected_cabin(self):
        reference = json.dumps(
            {
                "tile_index": 2,
                "evidence_class": "structured",
                "visual_complexity": "complex",
                "canonical_objects": [
                    {
                        "id": "house",
                        "identity": "wooden cabin",
                        "target_prompt": "wooden cabin with pitched roof and windows",
                        "spatial_overlap": 0.2,
                    }
                ],
            }
        )
        water_only = json.dumps(
            {
                "local_caption": "calm water with ripples",
                "dominant_region": "water",
                "visible_boundaries": "",
                "object_id": "",
                "object_prompt": "",
                "local_features": "",
                "corrections_applied": "",
                "target_prompt": "calm water with ripples",
            }
        )
        self.assertIn("omitted", _expected_object_problem(water_only, reference))
        self.assertIn(
            "omitted",
            SmartCachedTilePromptGenerator()._caption_problem(
                water_only,
                reference,
                {
                    "composition_mode": "task_directed",
                    "task_preset": "Upscale / Detailer",
                },
            ),
        )

        with_cabin = json.loads(water_only)
        with_cabin["local_caption"] = "wooden cabin and rocky shore above the water"
        with_cabin["target_prompt"] = with_cabin["local_caption"]
        self.assertEqual(
            _expected_object_problem(json.dumps(with_cabin), reference), ""
        )

        explicitly_absent = json.loads(water_only)
        explicitly_absent["corrections_applied"] = (
            "checked the wooden cabin candidate; it is not visible"
        )
        self.assertEqual(
            _expected_object_problem(json.dumps(explicitly_absent), reference), ""
        )

    def test_generic_building_identity_uses_concrete_canonical_aliases(self):
        reference = json.dumps(
            {
                "tile_index": 5,
                "tile_id": "T006",
                "evidence_class": "structured",
                "visual_complexity": "complex",
                "canonical_objects": [
                    {
                        "id": "object_1",
                        "identity": "building",
                        "target_prompt": (
                            "modern wooden cabin with dark roof, large glass windows, "
                            "and a balcony"
                        ),
                        "spatial_overlap": 0.2,
                    }
                ],
            }
        )
        cabin_caption = json.dumps(
            {
                "local_caption": "wooden cabin beside the lake",
                "dominant_region": "cabin",
                "visible_boundaries": "roof edge",
                "object_id": "",
                "object_prompt": "",
                "local_features": "balcony railing",
                "corrections_applied": "",
                "target_prompt": "wooden cabin with a roof and balcony",
            }
        )
        self.assertEqual(_expected_object_problem(cabin_caption, reference), "")
        selected = _select_canonical_object(
            json.loads(reference), json.loads(cabin_caption)
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["id"], "object_1")

        water_only = json.loads(cabin_caption)
        water_only.update(
            {
                "local_caption": "calm water with gentle ripples",
                "dominant_region": "water",
                "visible_boundaries": "",
                "local_features": "",
                "target_prompt": "calm water with gentle ripples",
            }
        )
        problem = _expected_object_problem(json.dumps(water_only), reference)
        self.assertIn("modern wooden cabin", problem)
        self.assertNotIn('candidate "building"', problem)

    def test_wholly_generic_object_is_not_a_hard_tile_blocker(self):
        reference = json.dumps(
            {
                "tile_index": 5,
                "evidence_class": "structured",
                "canonical_objects": [
                    {
                        "id": "object_1",
                        "identity": "building",
                        "target_prompt": "large dark building",
                        "spatial_overlap": 0.5,
                    }
                ],
            }
        )
        water_only = json.dumps(
            {
                "local_caption": "calm water with gentle ripples",
                "dominant_region": "water",
                "visible_boundaries": "",
                "object_id": "",
                "object_prompt": "",
                "local_features": "",
                "corrections_applied": "",
                "target_prompt": "calm water with gentle ripples",
            }
        )
        self.assertEqual(_expected_object_problem(water_only, reference), "")

    def test_cabin_omission_triggers_focused_retry_before_cache_write(self):
        class WaterThenCabinModel:
            def __init__(self):
                self.calls = 0
                self.prompts = []

            def tokenize(self, prompt, **kwargs):
                self.prompts.append(prompt)
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                return [self.calls]

            def decode(self, generated_ids):
                if self.calls == 1:
                    return json.dumps(
                        {
                            "local_caption": "calm water with ripples",
                            "dominant_region": "water",
                            "visible_boundaries": "",
                            "surface_id": "water",
                            "surface_prompt": "calm water",
                            "object_id": "",
                            "object_prompt": "",
                            "local_features": "",
                            "corrections_applied": "",
                            "target_prompt": "calm water with ripples",
                        }
                    )
                return json.dumps(
                    {
                        "local_caption": (
                            "wooden cabin with pitched roof and windows on a rocky "
                            "shore above calm water"
                        ),
                        "dominant_region": "wooden cabin and rocky shoreline",
                        "visible_boundaries": "roof, wall, shore, and water boundaries",
                        "surface_id": "water",
                        "surface_prompt": "calm water",
                        "object_id": "house",
                        "object_prompt": (
                            "wooden cabin with pitched roof and large windows"
                        ),
                        "local_features": "rocks and reflected window light",
                        "corrections_applied": "recover source-faithful detail",
                        "target_prompt": (
                            "wooden cabin with pitched roof and large windows on a "
                            "rocky shoreline above calm water"
                        ),
                    }
                )

        reference = json.dumps(
            {
                "tile_index": 2,
                "tile_id": "T003",
                "evidence_class": "structured",
                "visual_complexity": "complex",
                "canonical_objects": [
                    {
                        "id": "house",
                        "identity": "wooden cabin",
                        "target_prompt": (
                            "wooden cabin with pitched roof and large windows"
                        ),
                        "spatial_overlap": 0.2,
                    }
                ],
            }
        )
        model = WaterThenCabinModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch.dict(
                os.environ, {"SMART_UPSCALER_CACHE_DIR": cache_directory}
            ):
                result = SmartCachedTilePromptGenerator().generate(
                    model,
                    torch.rand(1, 32, 32, 3),
                    "Inspect this exact tile and return the required JSON.",
                    reference,
                    {
                        "prompt_strategy": "task_directed",
                        "composition_mode": "task_directed",
                        "operation_mode": "enhance",
                        "task_preset": "Upscale / Detailer",
                        "edit_action": (
                            "Upscale this image with realistic, source-faithful detail"
                        ),
                        "prompt_format": "instruction_edit",
                    },
                    "Managed by Prompt Director (recommended)",
                    "refresh",
                    "cabin-coverage-test",
                    "artifacts, seams",
                )
        self.assertEqual(model.calls, 2)
        self.assertIn("wooden cabin", result[0].casefold())
        self.assertIn("RETRY WRITE", result[4])
        self.assertIn("explicitly confirm", model.prompts[1])

    def test_unconfirmed_object_hypothesis_never_aborts_or_reaches_sampler(self):
        class AlwaysWaterModel:
            def __init__(self):
                self.calls = 0

            def tokenize(self, prompt, **kwargs):
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                return [self.calls]

            def decode(self, generated_ids):
                return json.dumps(
                    {
                        "local_caption": "calm water with gentle ripples",
                        "dominant_region": "water",
                        "visible_boundaries": "",
                        "surface_id": "water",
                        "surface_prompt": "calm water",
                        "object_id": "",
                        "object_prompt": "",
                        "local_features": "",
                        "corrections_applied": "",
                        "target_prompt": "calm water with gentle ripples",
                    }
                )

        reference = json.dumps(
            {
                "tile_index": 5,
                "tile_id": "T006",
                "evidence_class": "structured",
                "visual_complexity": "complex",
                "canonical_objects": [
                    {
                        "id": "object_1",
                        "identity": "building",
                        "target_prompt": (
                            "modern wooden cabin with dark roof, large glass windows, "
                            "and a balcony"
                        ),
                        "spatial_overlap": 0.2,
                    }
                ],
            }
        )
        model = AlwaysWaterModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch.dict(
                os.environ, {"SMART_UPSCALER_CACHE_DIR": cache_directory}
            ):
                result = SmartCachedTilePromptGenerator().generate(
                    model,
                    torch.rand(1, 32, 32, 3),
                    "Inspect this exact tile and return the required JSON.",
                    reference,
                    {
                        "prompt_strategy": "task_directed",
                        "composition_mode": "task_directed",
                        "operation_mode": "enhance",
                        "task_preset": "Upscale / Detailer",
                        "edit_action": (
                            "Upscale this image with realistic, source-faithful detail"
                        ),
                        "prompt_format": "instruction_edit",
                    },
                    "Managed by Prompt Director (recommended)",
                    "refresh",
                    "unconfirmed-cabin-does-not-abort",
                    "artifacts, seams",
                )
        self.assertEqual(model.calls, 3)
        self.assertIn("CONSERVATIVE-FALLBACK", result[4])
        self.assertNotIn("cabin", result[0].casefold())
        self.assertNotIn("building", result[0].casefold())

    def test_prompt_director_exposes_color_policy(self):
        _instruction, prompt_system, _combined = SmartUnifiedPromptGuidance().build(
            tile_colors="No color words (keep original colors)"
        )
        self.assertEqual(prompt_system["color_words"], "strip")

    def test_color_word_guard_strips_unrequested_colors(self):
        result = _strip_color_words(
            "calm deep teal water beneath a pale blue sky, fine detail"
        )
        self.assertNotIn("teal", result.casefold())
        self.assertNotIn("blue", result.casefold())
        self.assertIn("water", result.casefold())

    def test_color_word_guard_keeps_user_requested_family(self):
        result = _strip_color_words(
            "blue fabric beside green foliage", keep_sources=("make the fabric blue",)
        )
        self.assertIn("blue", result.casefold())
        self.assertNotIn("green", result.casefold())

    def test_resolver_strips_canonical_color_after_prompt_assembly(self):
        surface = {
            "id": "surface_1",
            "identity": "open water",
            "target_prompt": "calm deep teal open water",
        }
        reference = {
            "tile_index": 0,
            "evidence_class": "uniform",
            "visual_complexity": "simple",
            "canonical_surfaces": [surface],
        }
        caption = {
            "local_caption": "open water",
            "dominant_region": "open water",
            "visible_boundaries": "",
            "surface_id": "surface_1",
            "surface_prompt": surface["target_prompt"],
            "object_id": "",
            "object_prompt": "",
            "local_features": "",
            "corrections_applied": "",
            "target_prompt": surface["target_prompt"],
        }
        positive, _negative, _audit, _reference = SmartTilePromptResolver().resolve(
            json.dumps(caption),
            json.dumps(reference),
            {
                "prompt_strategy": "task_directed",
                "operation_mode": "enhance",
                "composition_mode": "task_directed",
                "task_preset": "Google Image Enhance",
                "edit_action": "Upscale this image with realistic detail",
                "prompt_format": "instruction_edit",
                "color_words": "strip",
            },
            "",
        )
        self.assertNotIn("teal", positive.casefold())
        self.assertIn("open water", positive.casefold())

    def test_material_requires_a_location(self):
        payload = {
            "surface_map": [],
            "material_map": [
                {
                    "id": "fabric",
                    "locations": [],
                    "identity": "woven fabric",
                    "target_prompt": "woven fabric",
                }
            ],
            "object_map": [],
        }
        problem = _global_caption_problem(
            json.dumps(payload), {"unified_instruction_ui": True}
        )
        self.assertIn("material_map", problem)
        self.assertIn("invalid locations", problem)

    def test_material_rejects_nearby_objects(self):
        payload = {
            "surface_map": [],
            "material_map": [
                {
                    "id": "stone",
                    "locations": ["center"],
                    "identity": "stone texture",
                    "target_prompt": "stone texture beside a bridge",
                }
            ],
            "object_map": [],
        }
        problem = _global_caption_problem(
            json.dumps(payload), {"unified_instruction_ui": True}
        )
        self.assertIn("material_map", problem)
        self.assertIn("only the surface itself", problem)

    def test_main_subject_does_not_mutate_an_unrelated_first_object(self):
        original = {
            "surface_map": [],
            "material_map": [],
            "object_map": [
                {
                    "id": "bridge",
                    "locations": ["center"],
                    "identity": "stone bridge",
                    "target_prompt": "stone bridge",
                }
            ],
        }
        generator = SmartCachedTextGenerate()
        result = json.loads(
            generator._identify_main_subject(
                json.dumps(original),
                {"unified_instruction_ui": True},
                lambda _question, response_limit=None: (
                    "person | top center: head; center: torso"
                ),
            )
        )
        self.assertEqual(result["object_map"][0]["identity"], "stone bridge")
        self.assertNotIn("parts", result["object_map"][0])
        self.assertEqual(result["object_map"][1]["identity"], "person")

    def test_skyscrapers_do_not_license_sky_wording(self):
        result = _strip_unsupported_atmosphere(
            "glass towers under a clear sky", evidence_text="dense skyscrapers"
        )
        self.assertNotIn("sky", result.casefold())


class CacheSafetyTests(unittest.TestCase):
    def test_model_fingerprint_hashes_more_than_four_sampled_values(self):
        first = torch.nn.Linear(16, 16, bias=False)
        second = torch.nn.Linear(16, 16, bias=False)
        second.load_state_dict(first.state_dict())
        with torch.no_grad():
            second.weight[5, 7] += 0.125
        self.assertNotEqual(_model_fingerprint(first), _model_fingerprint(second))

    def test_tensor_fingerprint_is_stable_and_content_sensitive(self):
        first = torch.arange(128, dtype=torch.float32).reshape(1, 8, 16, 1)
        second = first.clone()
        second[0, 3, 4, 0] += 1
        self.assertEqual(_tensor_fingerprint(first), _tensor_fingerprint(first.clone()))
        self.assertNotEqual(_tensor_fingerprint(first), _tensor_fingerprint(second))

    def test_vision_model_id_changes_cache_context(self):
        first = SmartCachedTextGenerate._context(
            1024, "Consistent caption", False, True, "", 1344, "model-a"
        )
        second = SmartCachedTextGenerate._context(
            1024, "Consistent caption", False, True, "", 1344, "model-b"
        )
        self.assertNotEqual(first, second)

    def test_prompt_cache_write_failure_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            blocker = Path(directory) / "not-a-directory"
            blocker.write_text("occupied", encoding="utf-8")
            with patch("nodes.cache._cache_root", return_value=blocker):
                path, error = _write_prompt("abc", "caption", "test")
        self.assertIsNone(path)
        self.assertTrue(error)


class CaptionCropTests(unittest.TestCase):
    def test_caption_image_excludes_sampler_padding(self):
        metadata = {
            "image_batch": 1,
            "image_width": 16,
            "image_height": 20,
            "tile_width": 64,
            "tile_height": 64,
            "scale_factor": 1.0,
            "output_tile_width": 64,
            "output_tile_height": 64,
            "tiles": [
                {
                    "tile_index": 0,
                    "source_index": 0,
                    "row": 0,
                    "column": 0,
                    "position": "center",
                    "x": 0,
                    "y": 0,
                    "width": 16,
                    "height": 20,
                }
            ],
        }
        tile = torch.rand(1, 64, 64, 3)
        tile_images, _instructions, _references, _seeds, caption_images = (
            SmartTileJobDirector().build(
                tile,
                json.dumps(metadata),
                None,
                {"prompt_strategy": "direct_user", "direct_prompt": "enhance"},
                "all_tiles",
                1,
                0,
                "fixed",
            )
        )
        self.assertEqual(tuple(tile_images[0].shape[1:3]), (64, 64))
        self.assertEqual(tuple(caption_images[0].shape[1:3]), (20, 16))


class WaterfallRunTests(unittest.TestCase):
    """Failures found in the live Z-Turbo waterfall run (2026-07-28)."""

    PROMPT_SYSTEM = {
        "task_preset": "Upscale / Detailer",
        "prompt_strategy": "task_directed",
        "composition_mode": "task_directed",
        "operation_mode": "faithful_upscale",
        "edit_action": "Upscale this image with realistic, source-faithful detail.",
        "direct_prompt": "Upscale this image with realistic, source-faithful detail.",
        "user_instruction": "",
        "prompt_format": "description",
        "prompt_suffix": "",
    }

    WATERFALL = {
        "id": "waterfall",
        "identity": "waterfall",
        "source_appearance": "",
        "target_prompt": "smooth, white water flowing over mossy rocks in a multi-tiered cascade",
    }

    # Toronto waterfront run, 2026-07-30. The whole-image pass placed the skyline
    # across the entire picture, so its overlap with the bottom-left water tiles
    # was near total - and overlap alone used to drop the evidence bar to one
    # token. The water tile then confirmed the skyline on "pattern" and "subtle",
    # and the Klein edit engine painted a glass building into the lake.
    SKYLINE = {
        "id": "skyline",
        "identity": "urban skyline with glass and concrete buildings",
        "source_appearance": "",
        "target_prompt": (
            "glass and concrete buildings with reflective facades, sharp edges, and "
            "uniform grid patterns of windows, lit by daylight with subtle "
            "reflections on glass surfaces"
        ),
    }

    def _resolve(self, payload, reference):
        positive, _negative, response, _ref = SmartTilePromptResolver().resolve(
            json.dumps(payload), json.dumps(reference), self.PROMPT_SYSTEM, "seams"
        )
        return positive, json.loads(response)

    def test_scene_words_cannot_confirm_an_object_placed_elsewhere(self):
        # The bottom-left cliff tile shares only "water" and "mossy" with the
        # waterfall's canonical wording, and the brief measured zero overlap
        # there. An edit engine would otherwise paint a whole waterfall in.
        elsewhere = dict(self.WATERFALL, spatial_overlap=0.0)
        self.assertIsNone(
            _select_canonical_object(
                {"canonical_objects": [elsewhere]},
                {
                    "object_id": "",
                    "local_caption": "Mossy cliff face with dense green vegetation, dark water below",
                    "dominant_region": "Mossy cliff face",
                },
            )
        )

    def test_located_object_still_confirms_from_tile_words(self):
        here = dict(self.WATERFALL, spatial_overlap=0.6)
        self.assertIsNotNone(
            _select_canonical_object(
                {"canonical_objects": [here]},
                {
                    "object_id": "",
                    "local_caption": "A waterfall cascades over mossy rocks",
                    "dominant_region": "waterfall",
                },
            )
        )

    def test_appearance_words_cannot_confirm_a_picture_wide_object(self):
        # Near-total overlap says only that the object spans the picture. The
        # tile's own words are water words; the two it shares with the skyline
        # ("pattern", "subtle") describe looks, not identity.
        everywhere = dict(self.SKYLINE, spatial_overlap=0.98)
        self.assertIsNone(
            _select_canonical_object(
                {"canonical_objects": [everywhere]},
                {
                    "object_id": "",
                    "local_caption": (
                        "Dark, rippling water surface with subtle, fine wave patterns "
                        "and minor ripples across the entire frame"
                    ),
                    "dominant_region": "water",
                },
            )
        )

    def test_part_hint_still_confirms_a_spanning_object_from_one_word(self):
        # The whole-image pass expecting a facade here is real evidence about
        # this tile, unlike overlap - so one distinctive word still confirms.
        expected_here = dict(
            self.SKYLINE, spatial_overlap=0.98, part_in_this_tile="glass facade"
        )
        self.assertIsNotNone(
            _select_canonical_object(
                {"canonical_objects": [expected_here]},
                {
                    "object_id": "",
                    "local_caption": "Glass facade of the skyscraper seen from above",
                    "dominant_region": "facade",
                },
            )
        )

    SKYLINE_SCENE = {
        "id": "main subject",
        "identity": "urban skyline with waterfront park",
        "source_appearance": "",
        "target_prompt": (
            "dense cluster of tall buildings with glass and steel facades, lit with "
            "warm yellow and white lights, surrounding a green park with trees and "
            "pathways, all under twilight sky"
        ),
        "spatial_overlap": 1.0,
        "part_in_this_tile": "waterfront park with trees and pathways",
    }
    WATER_TILE_ANSWER = {
        "object_id": "",
        "surface_id": "water",
        "local_caption": (
            "dark teal water with gentle ripples, reflecting city lights and showing "
            "subtle wave patterns under low ambient light"
        ),
        "dominant_region": "water",
    }

    def test_a_tile_that_says_surface_yes_object_no_is_believed(self):
        """Aerial cityscape: every open-water tile answered `surface_id: water,
        object_id: ""` and was still handed the entire city skyline, which Klein
        painted into the river as pasted-in blocks of buildings."""
        self.assertIsNone(
            _select_canonical_object(
                {"canonical_objects": [self.SKYLINE_SCENE]}, self.WATER_TILE_ANSWER
            )
        )

    def test_a_coincidental_word_cannot_stand_in_for_the_expected_part(self):
        """The only word the water tile shared with the skyline was "lights",
        and it came from the object's long target prompt - not from the part the
        brief said to look for ("waterfront park with trees and pathways")."""
        no_surface = dict(self.WATER_TILE_ANSWER, surface_id="")
        self.assertIsNone(
            _select_canonical_object(
                {"canonical_objects": [self.SKYLINE_SCENE]}, no_surface
            )
        )
        # The same tile that actually shows the named part still confirms it.
        sees_the_park = dict(
            no_surface,
            local_caption="waterfront park with trees and winding pathways",
            dominant_region="park",
        )
        self.assertIsNotNone(
            _select_canonical_object(
                {"canonical_objects": [self.SKYLINE_SCENE]}, sees_the_park
            )
        )

    def test_a_surface_tile_can_still_confirm_an_object_it_really_sees(self):
        """A bridge over water: the tile names both, so both are kept. Only
        silence about the object counts as "no"."""
        selected = _select_canonical_object(
            {"canonical_objects": [self.SKYLINE_SCENE]},
            dict(self.WATER_TILE_ANSWER, object_id="main subject"),
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["id"], "main subject")

    def test_explicit_object_id_remains_the_escape_hatch(self):
        elsewhere = dict(self.WATERFALL, spatial_overlap=0.0)
        selected = _select_canonical_object(
            {"canonical_objects": [elsewhere]},
            {"object_id": "waterfall", "local_caption": "a waterfall"},
        )
        self.assertEqual(selected["id"], "waterfall")

    def test_legacy_reference_without_overlap_keeps_token_confirmation(self):
        legacy = dict(self.WATERFALL)
        self.assertIsNotNone(
            _select_canonical_object(
                {"canonical_objects": [legacy]},
                {
                    "object_id": "",
                    "local_caption": "mossy rocks beside flowing water",
                    "dominant_region": "rocks",
                },
            )
        )

    def test_sentences_survive_when_no_guard_removed_anything(self):
        caption = (
            "A dense canopy of green leaves with a prominent dark branch. "
            "Sunlight filters through the foliage, creating bright patches and "
            "soft shadows. Background trees are blurred and out of focus."
        )
        positive, _audit = self._resolve(
            {
                "local_caption": caption,
                "dominant_region": "Canopy of green leaves",
                "target_prompt": caption,
            },
            {"tile_index": 1, "position": "top center", "evidence_class": "structured"},
        )
        self.assertIn("bright patches and soft shadows.", positive)
        self.assertIn("blurred and out of focus", positive)
        self.assertNotIn(", Sunlight", positive)

    def test_absence_clause_still_removed(self):
        positive, _audit = self._resolve(
            {
                "local_caption": "green moss on rock",
                "dominant_region": "moss",
                "target_prompt": "Green moss on wet rock. No people are visible.",
            },
            {"tile_index": 0, "position": "top left", "evidence_class": "structured"},
        )
        self.assertNotIn("people", positive)
        self.assertIn("Green moss on wet rock", positive)

    def test_tile_drops_its_own_whole_image_location(self):
        positive, audit = self._resolve(
            {
                "local_caption": "mossy rock edge in shallow water",
                "dominant_region": "shallow water",
                "target_prompt": (
                    "Shallow, calm, deep teal water with gentle ripples, bordered "
                    "by a mossy rock edge at bottom center."
                ),
            },
            {
                "tile_index": 7,
                "position": "bottom center",
                "evidence_class": "structured",
            },
        )
        self.assertNotIn("bottom center", positive)
        self.assertIn("mossy rock edge", positive)
        self.assertIn("whole-image location", audit["corrections_applied"])

    def test_other_in_tile_positions_are_untouched(self):
        positive, _audit = self._resolve(
            {
                "local_caption": "a dark branch",
                "dominant_region": "branch",
                "target_prompt": "A dark branch crossing the top left of the crop.",
            },
            {
                "tile_index": 7,
                "position": "bottom center",
                "evidence_class": "structured",
            },
        )
        self.assertIn("top left", positive)

    def test_denoise_prompt_drops_the_edge_cut_instruction(self):
        """"edge-cut features remain clipped at the same image edge" is a rule
        for an edit model. A denoise sampler paints every noun it is handed, so
        that phrase asks for thin cut fragments in a smooth sky - the stick-like
        debris the user reported."""
        payload = {
            "local_caption": "smooth pale sky",
            "dominant_region": "muted teal sky with soft, diffuse haze",
            "visible_boundaries": "the tile's lower edge meets a ridge line",
            "target_prompt": "muted teal sky with soft, diffuse haze",
        }
        reference = {
            "tile_index": 0,
            "position": "top left",
            "evidence_class": "sparse",
        }
        positive, _audit = self._resolve(payload, reference)
        self.assertNotIn("edge-cut", positive)
        self.assertIn("teal sky with soft, diffuse haze", positive)

        edit_system = dict(self.PROMPT_SYSTEM, prompt_format="instruction_edit")
        edit_positive, _neg, _resp, _ref = SmartTilePromptResolver().resolve(
            json.dumps(payload), json.dumps(reference), edit_system, "seams"
        )
        self.assertIn("edge-cut features remain clipped", edit_positive)

    def test_frame_only_clauses_go_but_real_content_stays(self):
        self.assertEqual(
            _strip_frame_talk(
                "muted teal sky with soft haze; edge-cut features remain "
                "clipped at the same image edge"
            ),
            "muted teal sky with soft haze",
        )
        # A clause naming something real survives even though it says "tile".
        kept = "The right edge of the tile cuts through a rocky outcrop"
        self.assertEqual(_strip_frame_talk(kept), kept)
        # Nothing to remove: the text comes back byte-identical.
        plain = "Dark blue water with gentle ripples and soft reflections"
        self.assertEqual(_strip_frame_talk(plain), plain)

    def test_partly_overlapping_surface_cannot_replace_a_tile(self):
        """The lake picture: the middle-right tile is a mountain slope, but the
        nine coarse areas overlap the tile grid, so it carried a 20% claim from
        BOTH sky and water. The water phrase replaced the whole mountain
        description, and the model painted ripples across rock."""
        payload = {
            "local_caption": "mist-shrouded mountain slope with evergreen trees",
            "dominant_region": "mountain slope",
            # The tile does see the lake at the foot of the slope, so it names
            # the surface honestly - that must not hand over the whole tile.
            "surface_id": "water",
            "surface_prompt": "dark blue water with gentle ripples",
            "target_prompt": (
                "A steep, mist-shrouded mountain slope covered in dense evergreen "
                "trees, with a dark blue lake at the base"
            ),
        }
        reference = {
            "tile_index": 5,
            "position": "middle right",
            "evidence_class": "sparse",
            "canonical_surfaces": [
                {
                    "id": "water",
                    "identity": "open water",
                    "target_prompt": "dark blue water with gentle ripples",
                    "spatial_overlap": 0.2006,
                }
            ],
        }
        positive, audit = self._resolve(payload, reference)
        self.assertIn("mountain slope", positive)
        self.assertIn("covers only part of this tile", audit["corrections_applied"])

    def test_a_surface_that_covers_the_tile_still_takes_over(self):
        """Cross-tile consistency is the whole point - a real water tile must
        still get the shared wording, or neighbouring tiles seam."""
        payload = {
            "local_caption": "dark water",
            "dominant_region": "dark water",
            "surface_id": "water",
            "surface_prompt": "dark blue water with gentle ripples",
            "target_prompt": "dark water surface with a rock edge at the corner",
        }
        reference = {
            "tile_index": 8,
            "position": "bottom right",
            "evidence_class": "sparse",
            "canonical_surfaces": [
                {
                    "id": "water",
                    "identity": "open water",
                    "target_prompt": "dark blue water with gentle ripples",
                    "spatial_overlap": 1.0,
                }
            ],
        }
        positive, _audit = self._resolve(payload, reference)
        self.assertIn("dark blue water with gentle ripples", positive)

    def test_measured_uniform_run_still_stamps_regardless_of_overlap(self):
        """A pixel-measured run exists precisely to reach tiles the model's own
        location list missed, so it carries no overlap score and must not be
        filtered by one."""
        measured = {
            "id": "water",
            "identity": "open water",
            "target_prompt": "dark blue water with gentle ripples",
            "selection_source": "measured_uniform_region",
        }
        self.assertTrue(_surface_covers_tile(measured))
        self.assertTrue(_surface_covers_tile(dict(measured, spatial_overlap=0.05)))
        # Legacy references predate overlap scores and keep their behaviour.
        self.assertTrue(
            _surface_covers_tile(
                {"id": "water", "target_prompt": "dark blue water with ripples"}
            )
        )

    def test_caption_token_budget_grows_with_the_detail_level(self):
        """At Maximum the tile writes ~100 words of target_prompt on top of five
        other JSON fields. A flat 256-token cap cut it off mid-word and the
        fragment reached the sampler, so more detail produced a worse prompt."""
        budget = SmartCachedTilePromptGenerator._caption_token_budget
        simple = budget({"caption_detail": "Simple"})
        adaptive = budget({"caption_detail": "Adaptive by Tile (recommended)"})
        maximum = budget({"caption_detail": "Maximum (every visible item)"})
        self.assertLess(simple, adaptive)
        self.assertLess(adaptive, maximum)
        self.assertGreaterEqual(maximum, 768)
        # Unknown or missing values fall back to the recommended level.
        self.assertEqual(budget({}), adaptive)
        self.assertEqual(budget(None), adaptive)

    def test_caption_budget_is_part_of_the_cache_key(self):
        """Changing the detail level changes the answer, so it must change the
        key - and the lazy check must compute the SAME key as generation, or
        every cached tile misses and the vision model reloads."""
        generator = SmartCachedTilePromptGenerator()
        reference = json.dumps({"tile_index": 0, "evidence_class": "structured"})
        simple = generator._key_context(
            reference, "Consistent caption", 1344, "m", {"caption_detail": "Simple"}
        )
        maximum = generator._key_context(
            reference,
            "Consistent caption",
            1344,
            "m",
            {"caption_detail": "Maximum (every visible item)"},
        )
        self.assertNotEqual(simple, maximum)
        self.assertEqual(json.loads(simple)["max_length"], 256)
        self.assertEqual(json.loads(maximum)["max_length"], 768)

    def test_two_surfaces_claiming_the_same_area_cancel_there(self):
        """The apartment picture: the window view was filed as `sky` across all
        nine areas AND `water` across six, so tiles of hardwood floor carried a
        full-strength water claim. One patch is water or sky, never both."""
        metadata = {
            "image_width": 90,
            "image_height": 90,
            "tile_width": 30,
            "tile_height": 30,
            "scale_factor": 1.0,
        }

        def tile_at(x, y):
            return {
                "tile_index": 0,
                "source_index": 0,
                "row": 0,
                "column": 0,
                "position": "x",
                "x": x,
                "y": y,
                "width": 30,
                "height": 30,
                "core_x": x,
                "core_y": y,
                "core_width": 30,
                "core_height": 30,
            }

        nine = [
            "top left", "top center", "top right",
            "middle left", "center", "middle right",
            "bottom left", "bottom center", "bottom right",
        ]
        brief = {
            "surface_map": [
                {
                    "id": "sky",
                    "identity": "open sky",
                    "target_prompt": "pale blue sky with soft haze",
                    "locations": nine,
                },
                {
                    "id": "water",
                    "identity": "open water",
                    "target_prompt": "dark blue water with gentle ripples",
                    "locations": nine[3:],
                },
            ]
        }
        # Bottom row: contested by both, so neither reaches the tile.
        self.assertEqual(_surface_map_context(brief, tile_at(0, 60), metadata), [])
        # Top row: only sky ever claimed it, so it survives untouched.
        top = _surface_map_context(brief, tile_at(0, 0), metadata)
        self.assertEqual([entry["id"] for entry in top], ["sky"])

    def test_a_normal_landscape_split_is_untouched(self):
        """Sky above, water below, no overlap - the ordinary case must not
        change at all."""
        metadata = {
            "image_width": 90,
            "image_height": 90,
            "tile_width": 30,
            "tile_height": 30,
            "scale_factor": 1.0,
        }
        brief = {
            "surface_map": [
                {
                    "id": "sky",
                    "identity": "open sky",
                    "target_prompt": "pale blue sky with soft haze",
                    "locations": ["top left", "top center", "top right"],
                },
                {
                    "id": "water",
                    "identity": "open water",
                    "target_prompt": "dark blue water with gentle ripples",
                    "locations": ["bottom left", "bottom center", "bottom right"],
                },
            ]
        }
        bottom = {
            "tile_index": 0, "source_index": 0, "row": 0, "column": 0,
            "position": "bottom left", "x": 0, "y": 60, "width": 30, "height": 30,
            "core_x": 0, "core_y": 60, "core_width": 30, "core_height": 30,
        }
        self.assertEqual(
            [entry["id"] for entry in _surface_map_context(brief, bottom, metadata)],
            ["water"],
        )

    def test_false_detections_catch_singular_and_plural(self):
        """A user who bans "wires" means wire. Exact matching let every singular
        through: the wet-hair portrait banned `wires` and still shipped "a thin,
        translucent wire" to the sampler."""
        terms = _parse_false_detections("wood, sticks, twigs, wires")
        cleaned = _remove_false_detections(
            "A thin, translucent wire or cable runs diagonally", terms
        )
        self.assertNotIn("wire", cleaned)
        self.assertIn("cable", cleaned)
        # Compound debris is tidied rather than left dangling.
        self.assertEqual(
            _remove_false_detections("Out-of-focus dark wire-like structure", terms),
            "Out-of-focus dark like structure",
        )
        # Validation must agree, or a banned plural triggers no retry.
        self.assertIn(
            "wires", _excluded_detection_problem("a thin translucent wire", ["wires"])
        )
        # A whole canonical entry mentioning a banned concept is still dropped.
        self.assertTrue(_entry_contains_excluded("thin wire fencing", ["wires"]))

    def test_plural_forms_never_reach_a_different_word(self):
        """Only regular endings are generated - no stemming, no synonyms - so
        banning one word can never silently delete another."""
        forms = _false_detection_forms
        self.assertEqual(set(forms("wires")), {"wires", "wire"})
        self.assertEqual(set(forms("branch")), {"branch", "branches"})
        self.assertEqual(set(forms("glass")), {"glass", "glasses"})
        # "wood" must not reach "wooden", and "twigs" must not reach "branches".
        self.assertNotIn("wooden", forms("wood"))
        self.assertNotIn("branches", forms("twigs"))
        self.assertEqual(
            _remove_false_detections("a wooden cabin", ["wood"]), "a wooden cabin"
        )

    def test_presets_never_name_copyable_objects_as_examples(self):
        """Anything visible to the tile model gets echoed. The blur rule used to
        list "loose hair strands, wires, branches", and a wet-hair portrait came
        back with wire, cable and branches in half its tiles."""
        import preset_store

        for name, entry in preset_store.load_builtin_task_presets().items():
            text = str(entry.get("instructions", ""))
            with self.subTest(preset=name):
                for noun in ("wires", "branches", "twigs"):
                    self.assertNotIn(noun, text.casefold())
                # The rule itself must survive - it is what keeps flyaway hair
                # from being erased as background blur.
                if "blurred areas" in text:
                    self.assertIn("are content, not blur", text)

    def test_the_main_subject_is_demanded_even_inside_one_area(self):
        """object_map came back empty on four different real images because it
        was gated on "crosses areas" - and most subjects sit inside one ninth of
        the frame."""
        instruction = SmartUnifiedPromptGuidance().build()[0]
        self.assertIn("MAIN SUBJECT", instruction)
        self.assertIn("including when the whole of it sits inside a single area", instruction)
        self.assertNotIn(
            "one entry for EVERY main subject or large discrete thing that crosses areas",
            instruction,
        )

    def test_advisory_materials_survive_bare_identity_wording(self):
        """A 30-tile coastal villa returned a brief whose EVERY entry repeated its
        own identity ("rough white stone wall" -> "rough white stone wall").
        Judged by the authoritative surface rule that is thin, so every material
        was dropped and the stone path came back worded differently in each tile
        it crossed. Materials are advisory and cannot take a tile over, so that
        wording is worth sharing."""
        metadata = {
            "image_width": 90,
            "image_height": 90,
            "tile_width": 30,
            "tile_height": 30,
            "scale_factor": 1.0,
        }
        tile = {
            "tile_index": 0, "source_index": 0, "row": 2, "column": 0,
            "position": "bottom left", "x": 0, "y": 60, "width": 30, "height": 30,
            "core_x": 0, "core_y": 60, "core_width": 30, "core_height": 30,
        }
        brief = {
            "material_map": [
                {
                    "id": "stone path",
                    "identity": "uneven stone path",
                    "target_prompt": "uneven stone path",
                    "locations": ["bottom left"],
                },
                {
                    "id": "moss",
                    "identity": "moss",
                    "target_prompt": "moss",
                    "locations": ["bottom left"],
                },
            ]
        }
        offered = [e["id"] for e in _material_map_context(brief, tile, metadata)]
        # Multi-word wording is shared; a bare one-word answer still carries nothing.
        self.assertIn("stone path", offered)
        self.assertNotIn("moss", offered)

    def test_bare_surface_wording_is_still_refused(self):
        """The authoritative path is unchanged: a surface REPLACES a whole tile,
        so "water" -> "water" must never be stamped."""
        self.assertTrue(
            _map_prompt_is_thin(
                {"id": "wall", "identity": "rough white stone wall",
                 "target_prompt": "rough white stone wall"}
            )
        )
        self.assertFalse(
            _map_prompt_is_thin(
                {"id": "wall", "identity": "rough white stone wall",
                 "target_prompt": "rough white stone wall, deeply pitted and sun-bleached"}
            )
        )

    def test_unreadable_tile_falls_back_to_its_own_scene_context(self):
        """Roughly one tile in six of a dense 30-tile run cannot be read. The old
        fallback shipped "a sharp, naturally detailed photograph" - nothing for a
        denoise sampler to render. Everything used here was already selected for
        THIS tile by the whole-image pass and location filtering."""
        reference = {
            "tile_index": 18,
            "position": "bottom left",
            "evidence_class": "structured",
            "canonical_objects": [
                {
                    "id": "villa",
                    "identity": "white-walled coastal villa",
                    "target_prompt": "white-walled coastal villa",
                    "part_in_this_tile": "blue window; purple flowers",
                }
            ],
            "canonical_materials": [
                {
                    "id": "stone path",
                    "identity": "uneven stone path",
                    "target_prompt": "uneven stone path",
                }
            ],
        }
        text = _tile_context_fallback(reference)
        self.assertIn("blue window", text)
        self.assertIn("purple flowers", text)
        self.assertIn("uneven stone path", text)
        # The WHOLE subject is never pulled in - only the part expected here.
        self.assertNotIn("white-walled coastal villa", text)
        # With no context at all it stays empty so the caller keeps its own wording.
        self.assertEqual(_tile_context_fallback({"tile_index": 0}), "")
        self.assertEqual(_tile_context_fallback(None), "")

    def test_unreadable_water_tile_uses_its_measured_surface_not_a_part_hint(self):
        """T012 of the NYC aerial: open water carrying `water` at overlap 1.0 AND
        two objects whose part hints said "city skyline" and "parked boat". The
        first version of this fallback took the hints and Klein painted a city
        skyline into the East River. A measured surface that covers the tile is
        the only pixel-derived source here, so it wins outright and alone."""
        reference = {
            "position": "bottom right",
            "evidence_class": "structured",
            "canonical_surfaces": [
                {
                    "id": "water",
                    "identity": "open water",
                    "spatial_overlap": 1.0,
                    "target_prompt": "dark teal water with gentle ripples",
                }
            ],
            "canonical_objects": [
                {"id": "main subject", "spatial_overlap": 1.0, "part_in_this_tile": "city skyline"},
                {"id": "island park", "spatial_overlap": 1.0, "part_in_this_tile": "parked boat"},
            ],
        }
        text = _tile_context_fallback(reference)
        self.assertEqual(text, "dark teal water with gentle ripples")
        self.assertNotIn("skyline", text)
        self.assertNotIn("boat", text)

    def test_a_surface_clipping_the_tile_does_not_win_the_fallback(self):
        """Only a surface that actually covers the crop may stand in for it;
        otherwise the advisory sources are still the better answer."""
        reference = {
            "canonical_surfaces": [
                {"id": "water", "spatial_overlap": 0.2, "target_prompt": "dark teal water"}
            ],
            "canonical_objects": [{"id": "villa", "part_in_this_tile": "blue window"}],
        }
        self.assertEqual(_tile_context_fallback(reference), "blue window")

    def test_two_covering_surfaces_are_a_real_tie(self):
        reference = {
            "canonical_surfaces": [
                {"id": "sky", "spatial_overlap": 1.0, "target_prompt": "pale blue sky"},
                {"id": "water", "spatial_overlap": 1.0, "target_prompt": "dark teal water"},
            ],
            "canonical_materials": [{"id": "path", "target_prompt": "uneven stone path"}],
        }
        # Neither surface is picked arbitrarily; the advisory wording is used.
        self.assertEqual(_tile_context_fallback(reference), "uneven stone path")

    def test_a_surface_never_carries_what_it_reflects(self):
        """Night harbour: every open-water tile was stamped "reflecting city
        lights", so each one invented its own bright highlights and the tile grid
        became visible on the water. A surface phrase is painted onto tiles that
        hold ONLY that surface, so anything it is said to reflect gets painted
        there too - the same rule the contract already states for buildings."""
        strip = _strip_reflected_objects
        self.assertEqual(
            strip("dark teal water with gentle ripples and soft reflections of city lights"),
            "dark teal water with gentle ripples",
        )
        self.assertEqual(
            strip(
                "dark teal water with gentle ripples, reflecting city lights and "
                "showing subtle wave patterns under low ambient light"
            ),
            "dark teal water with gentle ripples, showing subtle wave patterns under low ambient light",
        )
        # A bare reflection is a real property of water and must survive.
        for safe in (
            "calm, deep teal water with gentle ripples and soft reflections",
            "muted teal sky with soft, diffuse haze",
        ):
            self.assertEqual(strip(safe), safe)

    def test_the_resolver_applies_the_reflection_strip(self):
        payload = {
            "local_caption": "open water",
            "dominant_region": "open water",
            "surface_id": "water",
            "surface_prompt": "dark teal water reflecting city lights",
            "target_prompt": "open water surface",
        }
        reference = {
            "tile_index": 7,
            "position": "bottom right",
            "evidence_class": "uniform",
            "canonical_surfaces": [
                {
                    "id": "water",
                    "identity": "open water",
                    "spatial_overlap": 1.0,
                    "target_prompt": "dark teal water with gentle ripples and reflections of city lights",
                }
            ],
        }
        positive, _audit = self._resolve(payload, reference)
        self.assertIn("dark teal water with gentle ripples", positive)
        self.assertNotIn("city lights", positive)

    def test_audit_log_surfaces_failed_tiles_at_the_top(self):
        """At 30+ tiles a failure is five lines buried in three pages."""
        tiles = [
            {"tile_id": "T001", "position": "top left", "cache_status": "WRITE | ab",
             "final_positive_prompt": "clear blue sky."},
            {"tile_id": "T014", "position": "middle left", "cache_status": "RETRY WRITE | cd",
             "final_positive_prompt": "an olive tree trunk."},
            {"tile_id": "T019", "position": "bottom left",
             "cache_status": "RETRY PLAIN-RECOVERY + CONSERVATIVE-FALLBACK-GUARD WRITE | ef",
             "final_positive_prompt": "blue window, purple flowers."},
        ]
        report = "\n".join(_tile_health_lines(tiles))
        self.assertIn("Tiles: 3 | read cleanly: 2", report)
        self.assertIn("NEEDS ATTENTION", report)
        self.assertIn("T019", report)
        self.assertIn("T014", report)  # re-asked once, listed separately
        self.assertNotIn("T001", report)
        # The flag also appears against the tile itself in the prompt list.
        self.assertTrue(_tile_needed_recovery(tiles[2]))
        self.assertFalse(_tile_needed_recovery(tiles[0]))
        self.assertFalse(_tile_needed_recovery(tiles[1]))
        self.assertEqual(_tile_health_lines([]), [])

    def test_map_entries_are_told_to_add_to_their_identity(self):
        """Every entry of a real 30-tile brief repeated its own identity while
        obeying the letter of "a single word is not a description"."""
        instruction = SmartUnifiedPromptGuidance().build()[0]
        self.assertEqual(instruction.count("It must say MORE than `identity`"), 3)
        self.assertNotIn("a single word is not a description", instruction)

    def test_material_candidate_repeating_its_own_name_is_dropped(self):
        metadata = {
            "image_width": 90,
            "image_height": 90,
            "tile_width": 30,
            "tile_height": 30,
            "scale_factor": 1.0,
        }
        tile = {
            "tile_index": 0,
            "source_index": 0,
            "row": 0,
            "column": 0,
            "position": "top left",
            "x": 0,
            "y": 0,
            "width": 30,
            "height": 30,
            "core_x": 0,
            "core_y": 0,
            "core_width": 30,
            "core_height": 30,
        }
        brief = {
            "material_map": [
                {
                    "id": "moss",
                    "identity": "moss",
                    "target_prompt": "moss",
                    "locations": ["top left"],
                },
                {
                    "id": "bark",
                    "identity": "bark",
                    "target_prompt": "coarse ridged grey-brown bark, medium pattern scale",
                    "locations": ["top left"],
                },
            ]
        }
        offered = [
            entry["id"] for entry in _material_map_context(brief, tile, metadata)
        ]
        self.assertEqual(offered, ["bark"])


if __name__ == "__main__":
    unittest.main()
