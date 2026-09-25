import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from nodes.blending import SmartTileBlender
from nodes.audit import SmartTilePromptAuditLog
from nodes.cache import (
    SmartCachedTextGenerate,
    SmartCachedTilePromptGenerator,
    _whole_image_context_review,
)
from nodes.fidelity import SmartTileColorMatch, SmartTileFidelityColorMatch
from nodes.finalize import SmartTileFinalizer, _cross_tile_overlap_consistency
from nodes.processing import (
    SmartSamplerTileSelector,
    SmartTileMergePartialBatch,
    SmartTileSeed,
)
from nodes.prompting import (
    SmartTilePromptResolver,
    _source_caption_problem,
)
from nodes.tiling import SmartAdaptiveTilePlanner, SmartTilePlanner
from nodes.universal_prompting import (
    SmartPromptGuidance,
    SmartTileJobDirector,
    SmartUnifiedPromptGuidance,
    TASK_PRESETS,
    _build_all_tile_jobs,
)
from nodes.upscaled_tiling import SmartUpscaledTilePlanner, UPSCALE_METHODS


class SmartUpscalerPipelineTests(unittest.TestCase):
    def test_release_workflow_uses_only_registered_pipeline_nodes(self):
        import nodes as node_registry

        workflow_path = (
            Path(__file__).resolve().parents[1]
            / "workflow"
            / "Smart-Upscaler-Z-Turbo-v2.json"
        )
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        smart_types = {
            node["type"]
            for node in workflow["nodes"]
            if str(node["type"]).startswith("Smart")
        }

        # The shipped workflow exercises every registered node, so a node that
        # stops being wired in is a real regression rather than a stale test.
        self.assertEqual(smart_types, set(node_registry.NODE_CLASS_MAPPINGS))
        self.assertEqual(
            set(node_registry.NODE_CLASS_MAPPINGS),
            set(node_registry.NODE_DISPLAY_NAME_MAPPINGS),
        )
        # The sampler stays a separate replaceable block: core ComfyUI loader
        # and sampler nodes, none of them owned by this pack.
        types = {node["type"] for node in workflow["nodes"]}
        self.assertIn("UNETLoader", types)
        self.assertTrue(
            types & {"KSampler", "SamplerCustomAdvanced"},
            "the release workflow must still drive a plain ComfyUI sampler",
        )

    def test_uniform_tiles_use_one_canonical_surface_prompt_and_drop_false_objects(self):
        _, prompt_system, _ = SmartUnifiedPromptGuidance().build()
        surface = {
            "id": "S1",
            "identity": "recognized inland lake water",
            "source_appearance": "deep blue water",
            "target_prompt": "recognized inland lake water with a consistent deep-blue surface and subtle natural ripples",
            "matching_locations": ["bottom left", "bottom center"],
        }
        positives = []
        for tile_id, evidence_class, generated in (
            (
                "T009",
                "uniform",
                {
                    "local_caption": "dark water with dock structures",
                    "dominant_region": "dark water",
                    "visible_boundaries": "",
                    "surface_id": "",
                    "surface_prompt": "",
                    "local_features": "dock structures",
                    "corrections_applied": "",
                    "target_prompt": "dark water with dock structures",
                },
            ),
            (
                "T010",
                "sparse",
                {
                    "local_caption": "blue water with a faint false line",
                    "dominant_region": "blue water",
                    "visible_boundaries": "faint line near the lower edge",
                    "surface_id": "S1",
                    "surface_prompt": "different dark water wording",
                    "local_features": "invented pier",
                    "corrections_applied": "",
                    "target_prompt": "different dark water with an invented pier",
                },
            ),
        ):
            positive, _, audit, _ = SmartTilePromptResolver().resolve(
                json.dumps(generated),
                json.dumps(
                    {
                        "tile_index": int(tile_id[1:]) - 1,
                        "tile_id": tile_id,
                        "evidence_class": evidence_class,
                        "canonical_surfaces": [surface],
                    }
                ),
                prompt_system,
                "artifacts, seams",
            )
            positives.append(positive)
            self.assertIn(surface["target_prompt"], positive)
            self.assertNotIn("dock", positive.lower())
            self.assertNotIn("pier", positive.lower())
            self.assertEqual(json.loads(audit)["canonical_surface_id"], "S1")
            self.assertTrue(json.loads(audit)["canonical_surface_override"])

        self.assertNotIn("aerial", positives[0].lower())
        # The body is exactly the canonical surface prompt; the user's Tile
        # Prompt Suffix (default "fine detail") rides at the very end.
        self.assertEqual(
            positives[0].split(". ", 1)[1].rstrip("."),
            f"{surface['target_prompt']}, Fine detail",
        )

    def test_uniform_tile_receives_surface_map_but_not_regional_object_prior(self):
        _, prompt_system, _ = SmartUnifiedPromptGuidance().build()
        metadata = {
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
        global_context = {
            "source_medium": "3D map rendering",
            "view": "high-angle aerial view",
            "regional_map": ["bottom left: water with dock structures"],
            "surface_map": [
                {
                    "id": "S1",
                    "locations": ["bottom left", "bottom center"],
                    "identity": "recognized inland lake water",
                    "source_appearance": "deep blue with mild tonal variation",
                    "target_prompt": "recognized inland lake water with a consistent deep-blue surface and subtle natural ripples",
                }
            ],
        }
        _, instructions, references = _build_all_tile_jobs(
            torch.full((1, 8, 16, 3), 0.25),
            json.dumps(metadata),
            json.dumps(global_context),
            prompt_system,
        )

        reference = json.loads(references[0])
        self.assertEqual(reference["canonical_surfaces"][0]["id"], "S1")
        self.assertIn("canonical_surface_candidates", instructions[0])
        self.assertNotIn("dock structures", instructions[0])

    def test_master_prompt_requests_geography_and_generic_canonical_surfaces(self):
        global_instruction, _, _ = SmartUnifiedPromptGuidance().build()

        self.assertIn("surface_map", global_instruction)
        self.assertIn("object_map", global_instruction)
        self.assertIn("city, place, or named region", global_instruction)
        self.assertIn("target_prompt", global_instruction)
        self.assertNotIn("Toronto", global_instruction)
        self.assertNotIn("Lake Ontario", global_instruction)

    def test_unified_prompt_blueprint_exposes_the_complete_prompt_lineage(self):
        global_instruction, prompt_system, blueprint = SmartUnifiedPromptGuidance().build(
            instructions="""TASK: Time of Day
PROMPT DETAIL: Detailed
IMAGE ANALYSIS: Whole image + exact tiles
GLOBAL CONTEXT: Verified names when locally visible

Build a factual full-image reference, then transform each exact tile consistently.""",
            user_request="Turn the image into a realistic night scene.",
            known_false_detections="excluded structure, false surface",
        )

        self.assertEqual(
            prompt_system["context_awareness"], "Recognized Names When Visible"
        )
        self.assertEqual(prompt_system["prompt_blueprint"], blueprint)
        self.assertIn("HOW YOUR PROMPTS ARE BUILT", blueprint)
        self.assertIn("Step 1", blueprint)
        self.assertIn("Step 2", blueprint)
        self.assertIn("Step 3", blueprint)
        self.assertIn("Step 4", blueprint)
        self.assertIn("excluded structure, false surface", blueprint)
        review = _whole_image_context_review(
            '{"identity_anchors":[]}', prompt_system
        )
        self.assertIn(blueprint, review)
        self.assertIn("GENERATED MASTER SCENE PROMPT", review)

    def test_complex_tile_detail_is_user_selectable_and_has_its_own_budget(self):
        # The Prompt Detail Level dropdown lives outside the instructions text.
        instructions = """TASK: Upscale / Detailer
IMAGE ANALYSIS: Whole image + exact tiles
GLOBAL CONTEXT: Verified names when locally visible

MASTER SCENE PASS: Build a factual spatial reference from the supplied image.

EXACT-TILE PASS: Use only exact-tile evidence and spatially matching master context."""
        _, prompt_system, _ = SmartUnifiedPromptGuidance().build(
            instructions=instructions,
            tile_detail="Complex",
        )
        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        tiles, _, _, metadata_json = SmartAdaptiveTilePlanner().plan(
            image,
            min_tile_size=8,
            max_tile_size=8,
            overlap=0,
            feather=0,
            scale_factor=1.0,
            divisible_by=4,
            padding_mode="edge",
        )
        _, tile_instructions, _, _, _ = SmartTileJobDirector().build(
            tiles,
            metadata_json,
            "{}",
            prompt_system,
            "single_tile",
            1,
            1,
            "fixed",
        )

        self.assertEqual(prompt_system["caption_detail"], "Complex")
        self.assertIn("stay under 56 words", tile_instructions[0])
        self.assertIn("Describe every important visible region", tile_instructions[0])

    def test_simple_guidance_and_sparse_guard_keep_klein_prompt_literal(self):
        global_instruction, prompt_system, combined = SmartPromptGuidance().build(
            task_preset="Time of Day",
            caption_detail="Detailed",
            user_request="Convert this image to night and make it more realistic.",
        )
        self.assertIn("the tiles use for context", global_instruction)
        self.assertIn("requested target illumination", global_instruction)
        self.assertNotIn("prompt_strategy", combined)
        self.assertEqual(prompt_system["composition_mode"], "task_directed")
        self.assertEqual(prompt_system["caption_detail"], "Detailed")
        self.assertEqual(prompt_system["task_preset"], "Time of Day")
        self.assertTrue(prompt_system["edit_action"].startswith("Change this image"))

        leaked_caption = json.dumps(
            {
                "local_caption": (
                    "Deep blue water under night sky, subtle glow, reflections, soft haze"
                ),
                "dominant_region": "deep blue water",
                "visible_boundaries": "a subtle continuous edge at the top",
                "corrections_applied": "apply consistent nighttime illumination",
                "target_prompt": (
                    "Deep blue water at night with subtle local reflections along the visible edge"
                ),
            }
        )
        reference = json.dumps(
            {"tile_index": 11, "tile_id": "T012", "evidence_class": "sparse"}
        )
        positive, negative, audit, _ = SmartTilePromptResolver().resolve(
            leaked_caption,
            reference,
            prompt_system,
            "changed geometry, artifacts, seams",
        )

        self.assertNotIn("sky", positive.lower())
        self.assertNotIn("haze", positive.lower())
        self.assertTrue(positive.startswith("Change this image"))
        self.assertIn("deep blue water", positive.lower())
        self.assertIn("visible edge", positive)
        self.assertIn("new regions", negative)
        self.assertIn("material substitution", negative)
        self.assertEqual(json.loads(audit)["evidence_guard"], "task_directed_sparse")

        structured_positive, structured_negative, _, _ = SmartTilePromptResolver().resolve(
            json.dumps(
                {
                    "local_caption": "White masonry beside reflective glass.",
                    "dominant_region": "white masonry facade",
                    "visible_boundaries": "glass boundary at right",
                    "corrections_applied": "apply nighttime lighting",
                    "target_prompt": (
                        "White masonry with visible unit joints beside reflective glass under realistic nighttime lighting"
                    ),
                }
            ),
            json.dumps({"tile_index": 0, "tile_id": "T001", "evidence_class": "structured"}),
            prompt_system,
            "changed geometry, artifacts, seams",
        )
        self.assertIn("White masonry with visible unit joints", structured_positive)
        self.assertIn("changed layout", structured_negative)

        uniform_positive, _, uniform_audit, _ = SmartTilePromptResolver().resolve(
            json.dumps(
                {
                    "local_caption": "dark blue water with unsupported boats",
                    "dominant_region": "dark blue water with subtle tonal variation",
                    "visible_boundaries": "",
                    "corrections_applied": "apply consistent night illumination",
                    "target_prompt": "Dark blue water with subtle tonal variation under nighttime lighting",
                }
            ),
            json.dumps({"tile_index": 8, "tile_id": "T009", "evidence_class": "uniform"}),
            prompt_system,
            "artifacts",
        )
        self.assertIn("dark blue water with subtle tonal variation", uniform_positive.lower())
        self.assertNotIn("boats", uniform_positive.lower())
        self.assertEqual(json.loads(uniform_audit)["evidence_guard"], "task_directed_uniform")

    def test_prompt_director_owns_adaptive_detail_context_and_false_detections(self):
        global_instruction, prompt_system, combined = SmartPromptGuidance().build(
            task_preset="Time of Day",
            caption_detail="Adaptive by Tile (recommended)",
            user_request="Convert to night.",
            context_awareness="Recognized Names When Visible",
            known_false_detections="sky, ocean; damaged building",
        )

        self.assertEqual(
            prompt_system["caption_detail"], "Adaptive by Tile (recommended)"
        )
        self.assertEqual(
            prompt_system["context_awareness"], "Recognized Names When Visible"
        )
        self.assertEqual(
            prompt_system["known_false_detections"],
            ["sky", "ocean", "damaged building"],
        )
        self.assertIn("never invent content", global_instruction)
        self.assertIn("scene_type", global_instruction)
        self.assertIn("object_map", global_instruction)
        self.assertIn("surface_map", global_instruction)
        self.assertIn("KNOWN FALSE DETECTIONS (VLM ONLY)", combined)

    def test_context_awareness_changes_only_supported_global_context(self):
        image = torch.rand((1, 16, 16, 3), dtype=torch.float32)
        tiles, _, _, metadata_json = SmartAdaptiveTilePlanner().plan(
            image,
            min_tile_size=16,
            max_tile_size=16,
            overlap=0,
            feather=0,
            scale_factor=1.0,
            divisible_by=4,
            padding_mode="edge",
        )
        global_context = json.dumps(
            {
                "scene_type": "historic European city",
                "geographic_context": "London, England",
                "view": "street-level view",
                "target_appearance": "invented dramatic glow",
                "surface_map": [],
                "object_map": [],
            }
        )

        def instruction_for(awareness):
            _, system, _ = SmartPromptGuidance().build(
                context_awareness=awareness,
                user_request="Improve clarity.",
            )
            return SmartTileJobDirector().build(
                tiles,
                metadata_json,
                global_context,
                system,
                "single_tile",
                1,
                1,
                "fixed",
            )[1][0]

        local_only = instruction_for("Local Evidence Only")
        scene_aware = instruction_for("Scene Type + Continuity (recommended)")
        recognized = instruction_for("Recognized Names When Visible")

        self.assertNotIn("historic European city", local_only)
        self.assertNotIn("London, England", local_only)
        self.assertIn(
            "the exact tile supplies all content vocabulary", local_only
        )
        self.assertIn("historic European city", scene_aware)
        # The place name reaches a tile only when names are asked for.
        self.assertNotIn("London, England", scene_aware)
        self.assertIn("London, England", recognized)
        self.assertIn("A verified name may be used only when", recognized)
        self.assertNotIn("invented dramatic glow", recognized)

    def test_adaptive_tile_detail_records_complexity_and_repeated_unit_scale(self):
        image = torch.rand((1, 32, 32, 3), dtype=torch.float32)
        tiles, _, _, metadata_json = SmartAdaptiveTilePlanner().plan(
            image,
            min_tile_size=32,
            max_tile_size=32,
            overlap=0,
            feather=0,
            scale_factor=1.0,
            divisible_by=4,
            padding_mode="edge",
        )
        _, system, _ = SmartPromptGuidance().build(
            caption_detail="Adaptive by Tile (recommended)",
            context_awareness="Scene Type + Continuity (recommended)",
        )
        outputs = SmartTileJobDirector().build(
            tiles,
            metadata_json,
            json.dumps(
                {
                    "scene_type": "dense forest",
                    "recurring_details": [
                        "dense clusters of many tiny leaves with close spacing"
                    ],
                }
            ),
            system,
            "single_tile",
            1,
            2,
            "fixed",
        )
        instruction = outputs[1][0]
        reference = json.loads(outputs[2][0])

        self.assertIn(reference["visual_complexity"], ("complex", "dense"))
        self.assertNotIn("many tiny leaves", instruction)
        self.assertIn("individual scale, density, spacing", instruction)
        self.assertIn("Never replace many small repeated elements", instruction)
        self.assertIn("coarse local position", instruction)
        self.assertIn("compare every noun and visual effect", instruction)
        self.assertIn("Copy the same supported", instruction)
        self.assertIn("Do not restate, decorate, or explain", instruction)

    def test_false_detection_is_retried_before_local_prompt_reaches_sampler(self):
        class FalseThenSafeCaptionModel:
            def __init__(self):
                self.calls = 0
                self.generate_args = []

            def tokenize(self, prompt, **kwargs):
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                self.generate_args.append(kwargs)
                return [self.calls]

            def decode(self, generated_ids):
                if self.calls == 1:
                    return json.dumps(
                        {
                            "local_caption": "many tiny leaves beneath a sky",
                            "dominant_region": "dense foliage",
                            "visible_boundaries": "branches cross the upper edge",
                            "corrections_applied": "convert the lighting to night",
                            "target_prompt": "many tiny leaves under a dark sky",
                        }
                    )
                return json.dumps(
                    {
                        "local_caption": "dense clusters of many tiny leaves",
                        "dominant_region": "dense foliage",
                        "visible_boundaries": "branches cross the upper edge",
                        "corrections_applied": "convert existing illumination to night",
                        "target_prompt": "dense clusters of many tiny leaves with subdued night illumination",
                    }
                )

        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        reference = json.dumps(
            {"tile_index": 0, "tile_id": "T001", "evidence_class": "structured"}
        )
        _, system, _ = SmartPromptGuidance().build(
            task_preset="Time of Day",
            user_request="Convert to night.",
            known_false_detections="sky",
        )
        model = FalseThenSafeCaptionModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch("nodes.cache._cache_root", return_value=Path(cache_directory)):
                result = SmartCachedTilePromptGenerator().generate(
                    model,
                    image,
                    "Inspect this exact tile.",
                    reference,
                    system,
                    "Managed by Prompt Director (recommended)",
                    "refresh",
                    "false-detection-local-test",
                    "artifacts, seams",
                )

        self.assertEqual(model.calls, 2)
        self.assertTrue(all(not args["do_sample"] for args in model.generate_args))
        self.assertNotIn("sky", result[0].lower())
        self.assertIn("RETRY WRITE", result[4])

    def test_false_detection_repeated_after_retry_is_removed_without_aborting(self):
        class RepeatsFalseDetectionModel:
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
                        "local_caption": "glass towers beside rocks",
                        "dominant_region": "urban glass towers",
                        "visible_boundaries": "tower edges cross the upper boundary",
                        "corrections_applied": "repair texture warping and remove rocks",
                        "target_prompt": "realistic glass towers with rocks",
                    }
                )

        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        reference = json.dumps(
            {
                "tile_index": 2,
                "tile_id": "T003",
                "evidence_class": "structured",
                "visual_complexity": "moderate",
            }
        )
        _, system, _ = SmartPromptGuidance().build(
            task_preset="Google Image Enhance",
            known_false_detections="rocks",
        )
        model = RepeatsFalseDetectionModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch("nodes.cache._cache_root", return_value=Path(cache_directory)):
                result = SmartCachedTilePromptGenerator().generate(
                    model,
                    image,
                    "Inspect this exact tile.",
                    reference,
                    system,
                    "Managed by Prompt Director (recommended)",
                    "refresh",
                    "false-detection-guard-test",
                    "artifacts, seams",
                )

        self.assertEqual(model.calls, 2)
        self.assertNotIn("rocks", result[0].lower())
        self.assertIn("glass towers", result[0].lower())
        self.assertIn("FALSE-DETECTION-GUARD WRITE", result[4])

    def test_false_detection_guard_repairs_truncated_json_after_retry(self):
        class TruncatedFalseDetectionModel:
            def __init__(self):
                self.calls = 0

            def tokenize(self, prompt, **kwargs):
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                return [self.calls]

            def decode(self, generated_ids):
                return (
                    '{"local_caption":"aerial water beside rocks",'
                    '"dominant_region":"water and rocks",'
                    '"visible_boundaries":"shoreline beside rocks",'
                    '"corrections_applied":"repair warped water",'
                    '"target_prompt":"natural aerial water beside rocks"'
                )

        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        reference = json.dumps(
            {
                "tile_index": 5,
                "tile_id": "T006",
                "evidence_class": "sparse",
                "visual_complexity": "simple",
                "view_context": "high-angle aerial view",
            }
        )
        _, system, _ = SmartPromptGuidance().build(
            task_preset="Google Image Enhance", known_false_detections="rocks"
        )
        model = TruncatedFalseDetectionModel()
        result = SmartCachedTilePromptGenerator().generate(
            model,
            image,
            "Inspect exact tile T006.",
            reference,
            system,
            "Managed by Prompt Director (recommended)",
            "bypass",
            "truncated-false-detection-guard-test",
            "artifacts, seams",
        )

        self.assertEqual(model.calls, 2)
        self.assertNotIn("rocks", result[0].lower())
        self.assertIn("aerial water", result[0].lower())
        self.assertIn("false-detection", result[4].lower())

    def test_false_detection_guard_cleans_list_valued_caption_fields(self):
        class ListFieldFalseDetectionModel:
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
                        "local_caption": ["high-angle aerial water", "rocks"],
                        "dominant_region": ["water", "rocky shoreline"],
                        "visible_boundaries": ["concrete edge", "rocks"],
                        "corrections_applied": ["repair warped water", "remove rocky artifacts"],
                        "target_prompt": ["natural aerial water", "rocks"],
                    }
                )

        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        reference = json.dumps(
            {
                "tile_index": 5,
                "tile_id": "T006",
                "evidence_class": "sparse",
                "visual_complexity": "simple",
                "view_context": "high-angle aerial view",
            }
        )
        _, system, _ = SmartPromptGuidance().build(
            task_preset="Google Image Enhance",
            known_false_detections="rocks, rocky, old",
        )
        model = ListFieldFalseDetectionModel()
        result = SmartCachedTilePromptGenerator().generate(
            model,
            image,
            "Inspect exact tile T006.",
            reference,
            system,
            "Managed by Prompt Director (recommended)",
            "bypass",
            "list-valued-false-detection-guard-test",
            "changed geometry, duplicated objects, artifacts, seams",
        )

        self.assertEqual(model.calls, 2)
        self.assertNotIn("rocks", result[0].lower())
        self.assertNotIn("rocky", result[0].lower())
        self.assertIn("aerial water", result[0].lower())
        self.assertEqual(
            result[1],
            "new objects, new regions, new boundaries, changed viewpoint, changed layout, "
            "duplicated objects, unrequested material substitution, inconsistent cross-tile "
            "appearance, changed geometry, artifacts, seams",
        )
        self.assertIn("false-detection", result[4].lower())

    def test_false_only_caption_uses_plain_recovery_instead_of_stopping_batch(self):
        class FalseOnlyThenLocalModel:
            def __init__(self):
                self.calls = 0

            def tokenize(self, prompt, **kwargs):
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                return [self.calls]

            def decode(self, generated_ids):
                if self.calls == 3:
                    return "high-angle aerial water with gentle surface variation"
                return json.dumps(
                    {
                        "local_caption": "rocks",
                        "dominant_region": "rocks",
                        "visible_boundaries": "rocks",
                        "corrections_applied": "",
                        "target_prompt": "rocks",
                    }
                )

        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        reference = json.dumps(
            {
                "tile_index": 5,
                "tile_id": "T006",
                "evidence_class": "uniform",
                "visual_complexity": "simple",
                "view_context": "high-angle aerial view",
            }
        )
        _, system, _ = SmartPromptGuidance().build(
            task_preset="Google Image Enhance", known_false_detections="rocks"
        )
        model = FalseOnlyThenLocalModel()
        result = SmartCachedTilePromptGenerator().generate(
            model,
            image,
            "Inspect exact tile T006.",
            reference,
            system,
            "Managed by Prompt Director (recommended)",
            "bypass",
            "false-only-plain-recovery-test",
            "artifacts, seams",
        )

        self.assertEqual(model.calls, 3)
        self.assertNotIn("rocks", result[0].lower())
        self.assertIn("aerial water", result[0].lower())
        self.assertIn("plain-recovery", result[4].lower())

    def test_uniform_google_background_word_becomes_safe_aerial_continuous_region(self):
        class BackgroundThenWaterModel:
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
                            "local_caption": "dark blue background",
                            "dominant_region": "dark blue background",
                            "visible_boundaries": "",
                            "corrections_applied": "repair texture warping",
                            "target_prompt": "dark blue background",
                        }
                    )
                return json.dumps(
                    {
                        "local_caption": "high-angle aerial view of calm dark blue water",
                        "dominant_region": "calm waterfront water",
                        "visible_boundaries": "",
                        "corrections_applied": "restore natural continuous water texture",
                        "target_prompt": "high-angle aerial view of calm blue water with natural surface texture",
                    }
                )

        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        image[..., 2] = 0.2
        tiles, _, _, metadata_json = SmartAdaptiveTilePlanner().plan(
            image,
            min_tile_size=8,
            max_tile_size=8,
            overlap=0,
            feather=0,
            scale_factor=1.0,
            divisible_by=4,
            padding_mode="edge",
        )
        _, system, _ = SmartPromptGuidance().build(task_preset="Google Image Enhance")
        global_context = json.dumps(
            {
                "scene_type": "urban waterfront",
                "geographic_context": "",
                "view": "high-angle aerial view",
                "surface_map": [],
                "object_map": [],
            }
        )
        selected_images, instructions, references, _, _ = SmartTileJobDirector().build(
            tiles,
            metadata_json,
            global_context,
            system,
            "single_tile",
            1,
            1,
            "fixed",
        )
        # A uniform tile receives camera geometry only, never the scene label.
        self.assertIn("high-angle aerial", instructions[0])
        self.assertNotIn("urban waterfront", instructions[0])
        self.assertEqual(json.loads(references[0])["view_context"], "high-angle aerial view")

        model = BackgroundThenWaterModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch("nodes.cache._cache_root", return_value=Path(cache_directory)):
                result = SmartCachedTilePromptGenerator().generate(
                    model,
                    selected_images[0],
                    instructions[0],
                    references[0],
                    system,
                    "Managed by Prompt Director (recommended)",
                    "refresh",
                    "uniform-water-background-retry-test",
                    "artifacts, seams",
                )

        # The "background" caption is rejected and retried into a concrete surface.
        self.assertEqual(model.calls, 2)
        self.assertIn("calm blue water", result[0].lower())
        self.assertNotIn("background", result[0].lower())
        self.assertTrue(result[0].startswith("Repair this exact Google Earth"))
        self.assertIn("RETRY WRITE", result[4])

    def test_missing_local_caption_after_retry_uses_exact_tile_schema_fields(self):
        class MissingLocalCaptionModel:
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
                        "dominant_region": "aerial waterfront water and concrete shoreline",
                        "visible_boundaries": "shoreline crosses the upper right",
                        "corrections_applied": "restore continuous water and shoreline texture",
                        "target_prompt": (
                            "high-angle aerial view of calm waterfront water beside a clean concrete shoreline"
                        ),
                    }
                )

        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        reference = json.dumps(
            {
                "tile_index": 9,
                "tile_id": "T010",
                "evidence_class": "sparse",
                "visual_complexity": "simple",
                "view_context": "high-angle aerial view",
            }
        )
        _, system, _ = SmartPromptGuidance().build(task_preset="Google Image Enhance")
        model = MissingLocalCaptionModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch("nodes.cache._cache_root", return_value=Path(cache_directory)):
                result = SmartCachedTilePromptGenerator().generate(
                    model,
                    image,
                    "Inspect this exact T010 tile.",
                    reference,
                    system,
                    "Managed by Prompt Director (recommended)",
                    "refresh",
                    "missing-local-caption-schema-guard-test",
                    "artifacts, seams",
                )

        self.assertEqual(model.calls, 2)
        self.assertIn("aerial waterfront water", json.loads(result[2])["local_source_evidence"])
        self.assertIn("calm waterfront water", result[0].lower())
        self.assertIn("SCHEMA-GUARD WRITE", result[4])

    def test_managed_and_legacy_consistent_labels_share_deterministic_cache_policy(self):
        tile_resolver = SmartCachedTilePromptGenerator()
        self.assertEqual(
            tile_resolver._key_context(
                '{"tile_id":"T010"}', "Managed by Prompt Director (recommended)"
            ),
            tile_resolver._key_context(
                '{"tile_id":"T010"}', "Consistent caption (recommended)"
            ),
        )
        self.assertEqual(
            SmartCachedTextGenerate._context(
                384,
                "Managed by Prompt Director (recommended)",
                False,
                True,
                "global",
            ),
            SmartCachedTextGenerate._context(
                384, "Consistent caption", False, True, "global"
            ),
        )
        self.assertNotEqual(
            tile_resolver._key_context(
                '{"tile_id":"T010"}', "Managed by Prompt Director (recommended)"
            ),
            tile_resolver._key_context('{"tile_id":"T010"}', "Varied wording"),
        )

    def test_missing_local_caption_uses_focused_plain_recovery_after_empty_json(self):
        class EmptySchemaModel:
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
                if self.calls == 3:
                    return "Caption: high-angle aerial waterfront water beside a concrete shoreline"
                return json.dumps(
                    {
                        "dominant_region": "",
                        "visible_boundaries": "",
                        "corrections_applied": "",
                        "target_prompt": "",
                    }
                )

        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        reference = json.dumps(
            {
                "tile_index": 9,
                "tile_id": "T010",
                "evidence_class": "sparse",
                "visual_complexity": "simple",
                "view_context": "high-angle aerial view",
            }
        )
        _, system, _ = SmartPromptGuidance().build(task_preset="Google Image Enhance")
        model = EmptySchemaModel()
        result = SmartCachedTilePromptGenerator().generate(
            model,
            image,
            "Inspect this exact T010 tile.",
            reference,
            system,
            "Managed by Prompt Director (recommended)",
            "bypass",
            "empty-schema-plain-recovery-test",
            "artifacts, seams",
        )

        self.assertEqual(model.calls, 3)
        self.assertIn("Inspect only the pixels", model.prompts[-1])
        self.assertIn("aerial waterfront water", result[0].lower())
        self.assertIn("plain-recovery", result[4].lower())

    def test_missing_local_caption_uses_conservative_fallback_when_model_returns_nothing(self):
        class AlwaysEmptyModel:
            def tokenize(self, prompt, **kwargs):
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                return [1]

            def decode(self, generated_ids):
                return ""

        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        reference = json.dumps(
            {"tile_index": 9, "tile_id": "T010", "evidence_class": "sparse"}
        )
        _, system, _ = SmartPromptGuidance().build(task_preset="Google Image Enhance")
        result = SmartCachedTilePromptGenerator().generate(
            AlwaysEmptyModel(),
            image,
            "Inspect this exact T010 tile.",
            reference,
            system,
            "Managed by Prompt Director (recommended)",
            "bypass",
            "empty-plain-recovery-conservative-test",
            "artifacts, seams",
        )

        self.assertIn("dominant source region", result[0].lower())
        self.assertIn("conservative-fallback", result[4].lower())

    def test_conservative_fallback_is_renderable_for_a_denoise_engine(self):
        """A denoise sampler gets the tile text as the WHOLE prompt, so the
        last-resort caption must not be a sentence about "the source structures
        visible in this exact tile" - that describes nothing to render."""

        class AlwaysEmptyModel:
            def tokenize(self, prompt, **kwargs):
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                return [1]

            def decode(self, generated_ids):
                return ""

        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        reference = json.dumps(
            {"tile_index": 3, "tile_id": "T004", "evidence_class": "structured"}
        )
        _, system, _ = SmartPromptGuidance().build(
            task_preset="Upscale / Detailer",
            sampler_prompt_style="Plain description (SDXL, Flux, denoise)",
        )
        self.assertEqual(system["prompt_format"], "description")
        result = SmartCachedTilePromptGenerator().generate(
            AlwaysEmptyModel(),
            image,
            "Inspect this exact T004 tile.",
            reference,
            system,
            "Managed by Prompt Director (recommended)",
            "bypass",
            "denoise-conservative-fallback-test",
            "artifacts, seams",
        )
        self.assertIn("conservative-fallback", result[4].lower())
        self.assertNotIn("in this exact tile", result[0].lower())
        self.assertIn("naturally detailed photograph", result[0].lower())

        # Edit engines keep the original wording: their command carries the task.
        _, edit_system, _ = SmartPromptGuidance().build(task_preset="Upscale / Detailer")
        edit_result = SmartCachedTilePromptGenerator().generate(
            AlwaysEmptyModel(),
            image,
            "Inspect this exact T004 tile.",
            reference,
            edit_system,
            "Managed by Prompt Director (recommended)",
            "bypass",
            "edit-conservative-fallback-test",
            "artifacts, seams",
        )
        self.assertIn("in this exact tile", edit_result[0].lower())

    def test_background_caption_routes_through_general_recovery_without_aborting(self):
        class BackgroundOnlyModel:
            def __init__(self):
                self.calls = 0

            def tokenize(self, prompt, **kwargs):
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                return [self.calls]

            def decode(self, generated_ids):
                if self.calls == 3:
                    return "dark blue background"
                return json.dumps(
                    {
                        "local_caption": "dark blue background",
                        "dominant_region": "dark blue background",
                        "visible_boundaries": "",
                        "corrections_applied": "",
                        "target_prompt": "natural dark blue background",
                    }
                )

        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        reference = json.dumps(
            {
                "tile_index": 2,
                "tile_id": "T003",
                "evidence_class": "structured",
                "visual_complexity": "moderate",
            }
        )
        _, system, _ = SmartPromptGuidance().build(task_preset="Google Image Enhance")
        model = BackgroundOnlyModel()
        result = SmartCachedTilePromptGenerator().generate(
            model,
            image,
            "Inspect exact tile T003.",
            reference,
            system,
            "Managed by Prompt Director (recommended)",
            "bypass",
            "background-general-recovery-test",
            "artifacts, seams",
        )

        self.assertEqual(model.calls, 3)
        self.assertNotIn("background", result[0].lower())
        self.assertIn("source structures", result[0].lower())
        self.assertIn("conservative-fallback", result[4].lower())

    def test_uniform_photographic_background_is_valid_and_becomes_safe_region_prompt(self):
        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        reference = json.dumps(
            {
                "tile_index": 2,
                "tile_id": "T003",
                "evidence_class": "uniform",
                "visual_complexity": "simple",
            }
        )
        _, system, _ = SmartPromptGuidance().build(task_preset="Upscale / Detailer")
        model_response = json.dumps(
            {
                "local_caption": "soft blurred pale background",
                "dominant_region": "soft blurred pale background",
                "visible_boundaries": "",
                "corrections_applied": "",
                "target_prompt": "soft blurred pale background",
            }
        )

        self.assertEqual(
            _source_caption_problem(model_response, "uniform", True, (), "simple"), ""
        )
        positive, _, _, _ = SmartTilePromptResolver().resolve(
            model_response, reference, system, "artifacts, seams"
        )
        self.assertNotIn("background", positive.lower())
        self.assertIn("locally visible uniform surface", positive.lower())

    def test_false_detection_is_retried_in_whole_image_brief(self):
        class FalseThenSafeGlobalModel:
            def __init__(self):
                self.calls = 0
                self.generate_args = []

            def tokenize(self, prompt, **kwargs):
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.calls += 1
                self.generate_args.append(kwargs)
                return [self.calls]

            def decode(self, generated_ids):
                if self.calls == 1:
                    return '{"scene_type":"forest beneath a night sky"}'
                return '{"scene_type":"dense forest","recurring_details":["many tiny leaves"]}'

        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        prompt, system, _ = SmartPromptGuidance().build(
            known_false_detections="sky"
        )
        model = FalseThenSafeGlobalModel()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch("nodes.cache._cache_root", return_value=Path(cache_directory)):
                result = SmartCachedTextGenerate().generate(
                    model,
                    image,
                    prompt,
                    256,
                    "Managed by Prompt Director (recommended)",
                    False,
                    True,
                    "refresh",
                    "false-detection-global-test",
                    prompt_system=system,
                )

        self.assertEqual(model.calls, 2)
        self.assertTrue(all(not args["do_sample"] for args in model.generate_args))
        self.assertNotIn("sky", result[0].lower())
        self.assertIn("RETRY WRITE", result[1])

    def test_simple_guidance_no_vlm_passes_the_exact_user_request(self):
        _, prompt_system, _ = SmartPromptGuidance().build(
            task_preset="Upscale / Detailer",
            workflow_instruction="Analyze carefully and preserve local evidence.",
            user_request="Correct distortion and make the image more realistic.",
            image_analysis="Use User Request only (no image analysis)",
        )
        self.assertEqual(prompt_system["prompt_strategy"], "direct_user")
        self.assertEqual(prompt_system["composition_mode"], "direct_user")
        positive, negative, audit, _ = SmartTilePromptResolver().resolve(
            None,
            json.dumps({"tile_index": 0, "tile_id": "T001", "evidence_class": "uniform"}),
            prompt_system,
            "artifacts, seams",
        )
        self.assertEqual(positive, "Correct distortion and make the image more realistic.")
        self.assertEqual(negative, "artifacts, seams")
        self.assertEqual(json.loads(audit)["strategy"], "direct_user")

    def test_google_earth_task_brief_repairs_source_artifacts_in_final_prompt(self):
        global_instruction, prompt_system, _ = SmartPromptGuidance().build(
            task_preset="Google Image Enhance",
            caption_detail="Detailed",
        )
        self.assertIn("Google Earth or 3D map imagery", global_instruction)
        self.assertIn("photogrammetry reconstruction errors", global_instruction)
        self.assertIn("reconstruction artifacts to repair", global_instruction)
        self.assertNotIn("Describe the source only", global_instruction)
        self.assertEqual(prompt_system["prompt_strategy"], "task_directed")
        self.assertEqual(prompt_system["composition_mode"], "task_directed")
        self.assertTrue(prompt_system["edit_action"].startswith("Repair this exact Google Earth"))

        positive, _, audit, _ = SmartTilePromptResolver().resolve(
            json.dumps(
                {
                    "local_caption": (
                        "Broken-looking multi-story building with stretched facade textures"
                    ),
                    "dominant_region": "multi-story building",
                    "visible_boundaries": "roofline and street edge",
                    "corrections_applied": (
                        "repair photogrammetry warping and restore aligned facade geometry"
                    ),
                    "target_prompt": (
                        "Intact multi-story building with aligned glass and concrete facades, "
                        "a coherent roofline, and realistic continuous materials"
                    ),
                }
            ),
            json.dumps(
                {"tile_index": 8, "tile_id": "T009", "evidence_class": "structured"}
            ),
            prompt_system,
            "artifacts, seams",
        )

        self.assertIn("Intact multi-story building", positive)
        self.assertTrue(positive.startswith("Repair this exact Google Earth"))
        self.assertNotIn("broken", positive.lower())
        self.assertNotIn("damaged", positive.lower())
        self.assertEqual(json.loads(audit)["evidence_guard"], "task_directed_exact")

    def test_sparse_google_earth_target_uses_global_region_without_sky_or_ocean(self):
        _, prompt_system, _ = SmartPromptGuidance().build()
        positive, _, audit, _ = SmartTilePromptResolver().resolve(
            json.dumps(
                {
                    "local_caption": "deep blue water with a partial pale boundary",
                    "dominant_region": "Lake Ontario waterfront water",
                    "visible_boundaries": "pale boundary at upper right",
                    "corrections_applied": "remove reconstruction lines and restore natural water texture",
                    "target_prompt": (
                        "Realistic Lake Ontario waterfront water with natural surface texture and "
                        "a clean pale boundary at the upper right"
                    ),
                }
            ),
            json.dumps(
                {"tile_index": 11, "tile_id": "T012", "evidence_class": "sparse"}
            ),
            prompt_system,
            "artifacts, seams",
        )

        self.assertIn("Lake Ontario", positive)
        self.assertTrue(positive.startswith("Repair this exact Google Earth"))
        self.assertNotIn("ocean", positive.lower())
        self.assertNotIn("sky", positive.lower())
        self.assertIn("edge-cut features remain clipped at the same image edge", positive)
        self.assertEqual(json.loads(audit)["evidence_guard"], "task_directed_sparse")

    def test_sparse_portrait_does_not_inject_scene_view_or_complete_edge_feature(self):
        _, prompt_system, _ = SmartPromptGuidance().build(task_preset="Upscale / Detailer")
        positive, _, audit, _ = SmartTilePromptResolver().resolve(
            json.dumps(
                {
                    "local_caption": "Partial lips at the top edge and chin below",
                    "dominant_region": "chin and lower face",
                    "visible_boundaries": "partial lips clipped by the top edge",
                    "corrections_applied": "",
                    "target_prompt": "Lips and chin with soft lighting and dark hair at the sides",
                }
            ),
            json.dumps(
                {
                    "tile_index": 7,
                    "tile_id": "T008",
                    "evidence_class": "sparse",
                    "view_context": "Close-up portrait, looking toward camera",
                }
            ),
            prompt_system,
            "artifacts, seams",
        )

        self.assertNotIn("Close-up portrait", positive)
        self.assertIn("edge-cut features remain clipped at the same image edge", positive)
        self.assertEqual(json.loads(audit)["evidence_guard"], "task_directed_sparse")

    def test_google_target_description_is_not_sent_as_a_new_aerial_scene(self):
        _, prompt_system, _ = SmartPromptGuidance().build(task_preset="Google Image Enhance")
        positive, _, audit, _ = SmartTilePromptResolver().resolve(
            json.dumps(
                {
                    "local_caption": "Curved arena facade with vertical brown panels.",
                    "dominant_region": "curved arena facade",
                    "visible_boundaries": "roofline at top and street edge below",
                    "corrections_applied": "repair stretched photogrammetry textures",
                    "target_prompt": (
                        "Realistic high-angle aerial photograph of Madison Square Garden's curved facade with "
                        "continuous brown panels, aligned supports, and natural material detail"
                    ),
                }
            ),
            json.dumps({"tile_index": 2, "tile_id": "T003", "evidence_class": "structured"}),
            prompt_system,
            "artifacts, seams",
        )

        self.assertTrue(positive.startswith("Repair this exact Google Earth"))
        self.assertNotIn("aerial photograph of", positive.lower())
        self.assertIn("Madison Square Garden's curved facade", positive)
        self.assertEqual(json.loads(audit)["target_detail"], (
            "Madison Square Garden's curved facade with continuous brown panels, aligned supports, "
            "and natural material detail"
        ))

    def test_local_task_brief_excludes_global_object_and_environment_inventory(self):
        torch.manual_seed(5)
        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        tiles, _, _, metadata_json = SmartAdaptiveTilePlanner().plan(
            image,
            min_tile_size=8,
            max_tile_size=8,
            overlap=0,
            feather=0,
            scale_factor=1.0,
            divisible_by=4,
            padding_mode="edge",
        )
        _, prompt_system, _ = SmartPromptGuidance().build(
            task_preset="Time of Day",
            user_request="Convert the scene to realistic night.",
        )
        # Extra descriptive keys a chatty model might add must never reach a tile;
        # only scene_type and view are passed through by default.
        global_context = json.dumps(
            {
                "scene_type": "waterfront city",
                "geographic_context": "Toronto, Canada",
                "view": "high-angle aerial view",
                "target_appearance": "dark sky above an ocean and illuminated marina",
                "global_corrections": "add skyline lights and marina reflections",
                "surface_map": [],
                "object_map": [],
            }
        )
        _, instructions, _, _, _ = SmartTileJobDirector().build(
            tiles,
            metadata_json,
            global_context,
            prompt_system,
            "single_tile",
            1,
            7,
            "fixed",
        )
        instruction = instructions[0].lower()

        self.assertIn("convert the scene to realistic night", instruction)
        self.assertNotIn("toronto", instruction)
        self.assertNotIn("dark sky", instruction)
        self.assertNotIn("ocean", instruction)
        self.assertNotIn("marina", instruction)
        self.assertNotIn("skyline", instruction)

    def test_switching_builtin_task_replaces_a_stale_builtin_instruction(self):
        google_instruction = TASK_PRESETS["Google Image Enhance"]["instruction"]
        _, prompt_system, _ = SmartPromptGuidance().build(
            task_preset="Time of Day",
            workflow_instruction=google_instruction,
            user_request="night",
            image_analysis="Analyze whole image + exact tiles",
        )

        self.assertIn("requested time of day", prompt_system["workflow_instruction"])
        self.assertNotIn("Google Earth", prompt_system["workflow_instruction"])

    def test_planner_and_blender_reconstruct_image_batch(self):
        torch.manual_seed(7)
        images = torch.rand((2, 7, 9, 3), dtype=torch.float32)
        tiles, _, masks, metadata_json = SmartTilePlanner().plan(
            images,
            tile_width=5,
            tile_height=4,
            overlap=2,
            feather=2,
            scale_factor=1.0,
            padding_mode="edge",
        )

        result = SmartTileBlender().blend(
            tiles,
            metadata_json,
            masks,
            scale_mode="infer_from_tile_size",
        )[0]

        self.assertEqual(result.shape, images.shape)
        self.assertTrue(torch.allclose(result, images, atol=1e-6))

    def test_zero_overlap_does_not_leave_blending_holes(self):
        image = torch.ones((1, 8, 8, 3), dtype=torch.float32)
        tiles, _, masks, metadata_json = SmartTilePlanner().plan(
            image,
            tile_width=4,
            tile_height=4,
            overlap=0,
            feather=3,
            scale_factor=1.0,
            padding_mode="edge",
        )

        result = SmartTileBlender().blend(
            tiles,
            metadata_json,
            masks,
            scale_mode="infer_from_tile_size",
        )[0]

        self.assertTrue(torch.equal(result, image))

    def test_upscaled_planner_returns_authoritative_output_scale_tiles(self):
        image = torch.rand((1, 8, 12, 3), dtype=torch.float32)
        fake_model = types.SimpleNamespace(scale=4.0)

        def fake_upscale(_model, source_tiles, target_width, target_height, _batch_size):
            return torch.nn.functional.interpolate(
                source_tiles.movedim(-1, 1),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)

        with tempfile.TemporaryDirectory() as cache_directory:
            with patch("nodes.cache._cache_root", return_value=Path(cache_directory)):
                with patch(
                    "nodes.upscaled_tiling._upscale_tiles_with_model",
                    side_effect=fake_upscale,
                ):
                    tiles, preview, full_image, masks, metadata_json, preflight = (
                        SmartUpscaledTilePlanner().plan_and_upscale(
                            image,
                            upscale_model=fake_model,
                            min_tile_size=8,
                            max_tile_size=16,
                            overlap=4,
                            feather=2,
                            scale_factor=2.0,
                            divisible_by=4,
                            upscale_batch_size=1,
                            padding_mode="edge",
                        )
                    )
                with patch(
                    "nodes.upscaled_tiling._upscale_tiles_with_model",
                    side_effect=AssertionError("ESRGAN should be loaded from disk cache"),
                ):
                    cached = SmartUpscaledTilePlanner().plan_and_upscale(
                        image,
                        upscale_model=fake_model,
                        min_tile_size=8,
                        max_tile_size=16,
                        overlap=4,
                        feather=2,
                        scale_factor=2.0,
                        divisible_by=4,
                        upscale_batch_size=1,
                        padding_mode="edge",
                    )
        metadata = json.loads(metadata_json)

        self.assertTrue(metadata["tiles_are_output_scale"])
        self.assertEqual(tiles.shape[1:3], (metadata["output_tile_height"], metadata["output_tile_width"]))
        self.assertEqual(full_image.shape, (1, 16, 24, 3))
        self.assertEqual(preview.shape, full_image.shape)
        self.assertEqual(masks.shape[0], metadata["tile_count"])
        self.assertIn("Persistent preprocessing cache: WRITE", preflight)
        self.assertIn("Persistent preprocessing cache: HIT", cached[5])
        self.assertTrue(torch.equal(tiles, cached[0]))

    def test_preprocessing_cache_write_failure_is_nonfatal(self):
        image = torch.rand((1, 8, 12, 3), dtype=torch.float32)
        fake_model = types.SimpleNamespace(scale=4.0)

        def fake_upscale(_model, source_tiles, target_width, target_height, _batch_size):
            return torch.nn.functional.interpolate(
                source_tiles.movedim(-1, 1),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)

        with patch(
            "nodes.upscaled_tiling._upscale_tiles_with_model",
            side_effect=fake_upscale,
        ), patch(
            "nodes.upscaled_tiling.save_esrgan_tiles",
            return_value=(None, "insufficient disk space after cache pruning"),
        ):
            tiles, _, full_image, _, metadata_json, preflight = (
                SmartUpscaledTilePlanner().plan_and_upscale(
                    image,
                    upscale_model=fake_model,
                    min_tile_size=8,
                    max_tile_size=16,
                    overlap=4,
                    feather=2,
                    scale_factor=2.0,
                    divisible_by=4,
                    upscale_batch_size=1,
                    padding_mode="edge",
                )
            )

        metadata = json.loads(metadata_json)
        self.assertEqual(full_image.shape, (1, 16, 24, 3))
        self.assertGreater(tiles.numel(), 0)
        self.assertIn("WRITE SKIPPED", metadata["preprocess_cache"]["status"])
        self.assertIn("WRITE SKIPPED", preflight)

    def test_preprocessing_cache_space_guard_removes_stale_temp_and_skips_safely(self):
        from nodes.cache import DEFAULT_ESRGAN_CACHE_MAX_GB, save_esrgan_tiles

        self.assertEqual(DEFAULT_ESRGAN_CACHE_MAX_GB, 2.0)
        tiles = torch.rand((1, 4, 4, 3), dtype=torch.float32)
        with tempfile.TemporaryDirectory() as cache_directory:
            esrgan_directory = Path(cache_directory) / "esrgan"
            esrgan_directory.mkdir(parents=True)
            stale = esrgan_directory / ".failed-write.tmp"
            stale.write_bytes(b"partial")
            disk_usage = types.SimpleNamespace(total=1024, used=1024, free=0)
            with patch.dict(
                os.environ,
                {
                    "SMART_UPSCALER_ESRGAN_CACHE_MAX_GB": "12",
                    "SMART_UPSCALER_CACHE_FREE_RESERVE_GB": "2",
                },
            ), patch(
                "nodes.cache._cache_root", return_value=Path(cache_directory)
            ), patch("nodes.cache.shutil.disk_usage", return_value=disk_usage):
                path, reason = save_esrgan_tiles("low-space", tiles)

        self.assertIsNone(path)
        self.assertIn("insufficient disk space", reason)
        self.assertFalse(stale.exists())

    def test_standard_resize_upscaling_needs_no_ai_model(self):
        image = torch.rand((1, 4, 6, 3), dtype=torch.float32)

        def fake_resize(source_tiles, target_width, target_height, method):
            self.assertEqual(method, "lanczos")
            return torch.nn.functional.interpolate(
                source_tiles.movedim(-1, 1),
                size=(target_height, target_width),
                mode="bicubic",
                align_corners=False,
            ).movedim(1, -1).clamp(0.0, 1.0)

        planner = SmartUpscaledTilePlanner()
        with patch(
            "nodes.upscaled_tiling._resize_tiles_without_model",
            side_effect=fake_resize,
        ), patch(
            "nodes.upscaled_tiling._upscale_tiles_with_model",
            side_effect=AssertionError("AI upscaler must remain bypassed"),
        ):
            tiles, _, full_image, _, metadata_json, preflight = planner.plan_and_upscale(
                image,
                min_tile_size=8,
                max_tile_size=16,
                overlap=4,
                feather=2,
                scale_factor=2.0,
                divisible_by=4,
                upscale_batch_size=1,
                padding_mode="edge",
                cache_mode="bypass",
                upscale_method=UPSCALE_METHODS[1],
                upscale_model=None,
            )
        metadata = json.loads(metadata_json)
        self.assertEqual(metadata["upscale_method"], UPSCALE_METHODS[1])
        self.assertEqual(full_image.shape, (1, 8, 12, 3))
        self.assertEqual(
            tiles.shape[1:3],
            (metadata["output_tile_height"], metadata["output_tile_width"]),
        )
        self.assertIn("Lanczos", preflight)
        self.assertEqual(
            planner.check_lazy_status(
                image, UPSCALE_METHODS[1], 8, 16, 4, 2, 2.0, 4, 1, "edge"
            ),
            [],
        )
        self.assertEqual(
            planner.check_lazy_status(
                image, UPSCALE_METHODS[0], 8, 16, 4, 2, 2.0, 4, 1, "edge"
            ),
            ["upscale_model"],
        )

    def test_combined_tile_caption_generator_is_deterministic_and_lazy_cached(self):
        class FakeCaptionModel:
            def __init__(self):
                self.tokenize_args = None
                self.generate_args = None

            def tokenize(self, prompt, **kwargs):
                self.tokenize_args = (prompt, kwargs)
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                self.generate_args = kwargs
                return [2]

            def decode(self, generated_ids):
                return json.dumps(
                    {
                        "local_caption": "white brick with narrow joints and dark window openings",
                        "dominant_region": "white brick facade",
                        "visible_boundaries": "window openings cross the right edge",
                        "corrections_applied": "recover crisp supported masonry detail",
                        "target_prompt": (
                            "Detailed realistic white brick with narrow joints and dark window openings"
                        ),
                    }
                )

        image = torch.rand((1, 4, 4, 3), dtype=torch.float32)
        reference = json.dumps(
            {"tile_index": 0, "tile_id": "T001", "evidence_class": "structured"}
        )
        _, system, _ = SmartPromptGuidance().build(
            task_preset="Upscale / Detailer",
            caption_detail="Detailed",
        )
        node = SmartCachedTilePromptGenerator()
        model = FakeCaptionModel()
        arguments = (
            image,
            "Caption exact tile T001.",
            reference,
            system,
            "Consistent caption (recommended)",
            "read_write",
            "test-local-caption-v1",
            "artifacts, seams",
        )

        with tempfile.TemporaryDirectory() as cache_directory:
            with patch("nodes.cache._cache_root", return_value=Path(cache_directory)):
                self.assertEqual(node.check_lazy_status(None, *arguments), ["clip"])
                first = node.generate(model, *arguments)
                # thinking=True is the plain chat template: no empty think block.
                self.assertTrue(model.tokenize_args[1]["thinking"])
                self.assertFalse(model.tokenize_args[1]["skip_template"])
                self.assertFalse(model.generate_args["do_sample"])
                self.assertIn("white brick with narrow joints", first[0])
                self.assertIn("WRITE", first[4])
                self.assertEqual(node.check_lazy_status(None, *arguments), [])
                cached = node.generate(None, *arguments)
                self.assertEqual(cached[0], first[0])
                self.assertIn("HIT", cached[4])

    def test_tile_caption_retries_placeholder_response_before_sampling(self):
        class RetryCaptionModel:
            def __init__(self):
                self.prompts = []

            def tokenize(self, prompt, **kwargs):
                self.prompts.append(prompt)
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                return [len(self.prompts)]

            def decode(self, generated_ids):
                if len(self.prompts) == 1:
                    return json.dumps(
                        {
                            "local_caption": "No visible content in this crop; only black background.",
                            "dominant_region": None,
                            "visible_boundaries": None,
                        }
                    )
                return json.dumps(
                    {
                        "local_caption": "White masonry tower with narrow window rows and a flat roof",
                        "dominant_region": "white masonry facade",
                        "visible_boundaries": "roofline crosses the upper edge",
                        "corrections_applied": "repair reconstruction warping",
                        "target_prompt": (
                            "Intact white masonry tower with narrow window rows, a coherent flat roof, and corrected geometry"
                        ),
                    }
                )

        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        reference = json.dumps(
            {"tile_index": 11, "tile_id": "T012", "evidence_class": "structured"}
        )
        _, system, _ = SmartPromptGuidance().build()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch("nodes.cache._cache_root", return_value=Path(cache_directory)):
                result = SmartCachedTilePromptGenerator().generate(
                    RetryCaptionModel(),
                    image,
                    "Caption exact source tile T012.",
                    reference,
                    system,
                    "Consistent caption (recommended)",
                    "refresh",
                    "test-local-caption-v16",
                    "artifacts, seams",
                )

        self.assertIn("Intact white masonry tower", result[0])
        self.assertNotIn("black background", result[0].lower())
        self.assertIn("RETRY WRITE", result[4])

    def test_tile_caption_uses_conservative_fallback_if_placeholder_recovery_fails(self):
        class EmptyCaptionModel:
            def tokenize(self, prompt, **kwargs):
                return {"tokens": [1]}

            def generate(self, tokens, **kwargs):
                return [1]

            def decode(self, generated_ids):
                return '{"local_caption":"No visible content in this crop; only black background."}'

        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)
        reference = json.dumps(
            {"tile_index": 11, "tile_id": "T012", "evidence_class": "structured"}
        )
        _, system, _ = SmartPromptGuidance().build()
        result = SmartCachedTilePromptGenerator().generate(
            EmptyCaptionModel(),
            image,
            "Caption exact source tile T012.",
            reference,
            system,
            "Consistent caption (recommended)",
            "bypass",
            "test-local-caption-v16",
            "artifacts, seams",
        )

        self.assertNotIn("black background", result[0].lower())
        self.assertIn("source structures", result[0].lower())
        self.assertIn("conservative-fallback", result[4].lower())

    def test_truncated_json_uses_closed_local_caption_not_raw_json(self):
        _, system, _ = SmartPromptGuidance().build()
        truncated = (
            '{"local_caption":"White brick facade with narrow horizontal window rows",'
            '"dominant_region":"white brick facade",'
            '"corrections_applied":"repair warped joints",'
            '"target_prompt":"Intact white brick facade with straight narrow horizontal window rows",'
            '"visible_boundaries":"roofline at upper edge'
        )
        positive, _, audit, _ = SmartTilePromptResolver().resolve(
            truncated,
            json.dumps(
                {"tile_index": 11, "tile_id": "T012", "evidence_class": "structured"}
            ),
            system,
            "artifacts, seams",
        )

        self.assertTrue(positive.startswith("Repair this exact Google Earth"))
        self.assertTrue(
            positive.endswith(
                "Intact white brick facade with straight narrow horizontal window rows."
            )
        )
        self.assertFalse(positive.startswith("{"))
        self.assertEqual(
            json.loads(audit)["local_source_evidence"],
            "White brick facade with narrow horizontal window rows",
        )

    def test_fidelity_guard_suppresses_unsupported_coarse_objects(self):
        source = torch.full((1, 128, 128, 3), 0.35, dtype=torch.float32)
        generated = source.clone()
        generated[:, 32:96, 48:112, :] = 0.9

        corrected = SmartTileFidelityColorMatch().apply(
            generated,
            source,
            structure_preservation=100,
            detail_support="conservative",
            color_match_method="none",
            color_match_strength=0,
        )[0]

        generated_error = (generated - source).abs().mean()
        corrected_error = (corrected - source).abs().mean()
        self.assertLess(corrected_error, generated_error * 0.1)

    def test_tile_color_match_strength_and_modes(self):
        source = torch.zeros((1, 32, 32, 3), dtype=torch.float32)
        source[..., 0] = 0.2
        source[..., 1] = 0.4
        source[..., 2] = 0.7
        generated = torch.full_like(source, 0.8)
        node = SmartTileFidelityColorMatch()

        unchanged = node.apply(
            generated,
            source,
            structure_preservation=0,
            detail_support="balanced",
            color_match_method="rgb_mean",
            color_match_strength=0,
        )[0]
        matched = node.apply(
            generated,
            source,
            structure_preservation=0,
            detail_support="balanced",
            color_match_method="rgb_mean",
            color_match_strength=100,
        )[0]

        self.assertTrue(torch.equal(unchanged, generated))
        self.assertTrue(
            torch.allclose(
                matched.mean(dim=(1, 2)),
                source.mean(dim=(1, 2)),
                atol=1e-5,
            )
        )

    def test_standalone_tile_color_match_has_independent_control(self):
        source = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
        source[..., 1] = 0.5
        generated = torch.full_like(source, 0.9)
        node = SmartTileColorMatch()

        unchanged = node.apply(generated, source, "none", 100)[0]
        matched = node.apply(generated, source, "rgb_mean", 100)[0]

        self.assertTrue(torch.equal(unchanged, generated))
        self.assertTrue(
            torch.allclose(
                matched.mean(dim=(1, 2)),
                source.mean(dim=(1, 2)),
                atol=1e-5,
            )
        )

    def test_structure_only_fidelity_does_not_restore_source_brightness(self):
        source = torch.full((1, 64, 64, 3), 0.9, dtype=torch.float32)
        generated = torch.full((1, 64, 64, 3), 0.1, dtype=torch.float32)
        corrected = SmartTileFidelityColorMatch().apply(
            generated,
            source,
            structure_preservation=100,
            detail_support="balanced",
            color_match_method="none",
            color_match_strength=0,
            reference_mode="structure_only",
        )[0]

        self.assertLess(corrected.mean().item(), 0.2)

    def test_finalizer_combines_correction_partial_merge_and_blending(self):
        image = torch.zeros((1, 8, 12, 3), dtype=torch.float32)
        tiles, _, masks, metadata_json = SmartAdaptiveTilePlanner().plan(
            image,
            min_tile_size=8,
            max_tile_size=16,
            overlap=4,
            feather=2,
            scale_factor=1.0,
            divisible_by=4,
            padding_mode="edge",
        )
        metadata = json.loads(metadata_json)
        reference = json.dumps(metadata["tiles"][0])
        processed = torch.ones_like(tiles[0:1])

        corrected, final_image = SmartTileFinalizer().finalize(
            [processed],
            [reference],
            [tiles],
            [metadata_json],
            [masks],
            ["structure_only"],
            [0],
            ["balanced"],
            ["bilinear"],
        )

        self.assertEqual(len(corrected), 1)
        self.assertTrue(torch.equal(corrected[0], processed))
        self.assertEqual(final_image.shape, image.shape)
        self.assertGreater(final_image.mean().item(), 0.0)

    def test_cross_tile_consistency_aligns_neighbor_overlap_without_source_matching(self):
        tiles = torch.stack(
            (
                torch.full((4, 4, 3), 0.20, dtype=torch.float32),
                torch.full((4, 4, 3), 0.40, dtype=torch.float32),
            ),
            dim=0,
        )
        metadata = {
            "scale_factor": 1.0,
            "tiles": [
                {
                    "tile_index": 0,
                    "source_index": 0,
                    "row": 0,
                    "column": 0,
                    "x": 0,
                    "y": 0,
                    "width": 4,
                    "height": 4,
                },
                {
                    "tile_index": 1,
                    "source_index": 0,
                    "row": 0,
                    "column": 1,
                    "x": 2,
                    "y": 0,
                    "width": 4,
                    "height": 4,
                },
            ],
        }

        corrected = _cross_tile_overlap_consistency(tiles, metadata, 100)
        original_gap = (tiles[0, :, 2:4] - tiles[1, :, 0:2]).abs().mean()
        corrected_gap = (corrected[0, :, 2:4] - corrected[1, :, 0:2]).abs().mean()

        # The safety clamp deliberately limits correction to 8% per tile, so
        # even a deliberately extreme 20% mismatch is reduced, not erased.
        self.assertLess(corrected_gap, original_gap * 0.25)
        self.assertTrue(
            torch.allclose(corrected.mean(), tiles.mean(), atol=1e-6)
        )
        self.assertTrue(
            torch.equal(_cross_tile_overlap_consistency(tiles, metadata, 0), tiles)
        )

    def test_adaptive_planner_uses_even_core_grid_and_divisible_output_tiles(self):
        image = torch.ones((1, 104, 160, 3), dtype=torch.float32)
        tiles, preview, masks, metadata_json = SmartAdaptiveTilePlanner().plan(
            image,
            min_tile_size=64,
            max_tile_size=96,
            overlap=16,
            feather=8,
            scale_factor=2.0,
            divisible_by=16,
            padding_mode="edge",
        )
        metadata = json.loads(metadata_json)
        scaled_tiles = torch.nn.functional.interpolate(
            tiles.movedim(-1, 1),
            size=(
                int(metadata["output_tile_height"]),
                int(metadata["output_tile_width"]),
            ),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).movedim(1, -1).clamp(0.0, 1.0)
        result = SmartTileBlender().blend(
            scaled_tiles,
            metadata_json,
            masks,
            scale_mode="metadata_scale_factor",
        )[0]

        self.assertEqual(metadata["planner"], "adaptive_even_grid")
        self.assertEqual(metadata["output_overlap"], 16)
        self.assertEqual(metadata["output_feather"], 8)
        self.assertEqual(metadata["output_tile_width"] % 16, 0)
        self.assertEqual(metadata["output_tile_height"] % 16, 0)
        self.assertEqual(preview.shape[1:3], (208, 320))
        self.assertFalse(torch.equal(preview, torch.ones_like(preview)))
        self.assertEqual(result.shape, (1, 208, 320, 3))
        self.assertTrue(torch.allclose(result, torch.ones_like(result), atol=1e-6))

    def test_planner_always_rounds_sampler_tiles_to_a_multiple_of_32(self):
        # Qwen Image 2.1 rounds its own working size to 32 and hands the tile
        # back at that size, so anything less would not survive the stitcher.
        image = torch.zeros((1, 300, 420, 3), dtype=torch.float32)
        _, _, _, metadata_json = SmartAdaptiveTilePlanner().plan(
            image,
            min_tile_size=300,
            max_tile_size=520,
            overlap=24,
            feather=12,
            scale_factor=2.0,
            divisible_by=16,
            padding_mode="edge",
        )
        metadata = json.loads(metadata_json)

        self.assertEqual(metadata["divisible_by"], 32)
        self.assertEqual(metadata["output_tile_width"] % 32, 0)
        self.assertEqual(metadata["output_tile_height"] % 32, 0)

        _, _, _, aligned_json = SmartAdaptiveTilePlanner().plan(
            image,
            min_tile_size=300,
            max_tile_size=520,
            overlap=24,
            feather=12,
            scale_factor=2.0,
            divisible_by=64,
            padding_mode="edge",
        )
        aligned = json.loads(aligned_json)

        # A setting that is already 32-aligned keeps its own grid untouched.
        self.assertEqual(aligned["divisible_by"], 64)
        self.assertEqual(aligned["output_tile_width"] % 64, 0)
        self.assertEqual(aligned["output_tile_height"] % 64, 0)

    def test_partial_merge_replaces_selected_tile_and_fills_the_rest(self):
        image = torch.zeros((1, 32, 48, 3), dtype=torch.float32)
        source_tiles, _, _, metadata_json = SmartAdaptiveTilePlanner().plan(
            image,
            min_tile_size=32,
            max_tile_size=64,
            overlap=16,
            feather=8,
            scale_factor=2.0,
            divisible_by=4,
            padding_mode="edge",
        )
        metadata = json.loads(metadata_json)
        target_height = metadata["output_tile_height"]
        target_width = metadata["output_tile_width"]
        replacement = torch.ones((1, target_height, target_width, 3), dtype=torch.float32)
        reference = json.dumps(metadata["tiles"][1])

        merged = SmartTileMergePartialBatch().merge(
            [replacement],
            [reference],
            [source_tiles],
            [metadata_json],
            ["bilinear"],
        )[0]

        self.assertEqual(merged.shape[0], metadata["tile_count"])
        self.assertTrue(torch.equal(merged[1], torch.ones_like(merged[1])))
        self.assertTrue(torch.equal(merged[0], torch.zeros_like(merged[0])))

    def test_sampler_tile_selector_keeps_every_sampler_input_aligned(self):
        image_batch = torch.stack(
            (
                torch.full((4, 4, 3), 0.1),
                torch.full((4, 4, 3), 0.2),
                torch.full((4, 4, 3), 0.3),
            )
        )
        # Deliberately order T-numbers differently from list positions. The node
        # must select by the human T-number and then keep every paired value at
        # that same list position.
        references = [
            json.dumps({"tile_index": 2, "tile_id": "T003"}),
            json.dumps({"tile_index": 0, "tile_id": "T001"}),
            json.dumps({"tile_index": 1, "tile_id": "T002"}),
        ]
        selected = SmartSamplerTileSelector().select(
            [image_batch],
            ["positive T003", "positive T001", "positive T002"],
            ["negative T003", "negative T001", "negative T002"],
            references,
            [303, 101, 202],
            ["One tile test"],
            [2],
        )

        self.assertEqual([len(values) for values in selected], [1, 1, 1, 1, 1])
        self.assertTrue(
            torch.equal(selected[0][0], image_batch[2:3])
        )
        self.assertEqual(selected[1][0], "positive T002")
        self.assertEqual(selected[2][0], "negative T002")
        self.assertEqual(json.loads(selected[3][0])["tile_index"], 1)
        self.assertEqual(selected[4][0], 202)

    def test_sampler_tile_selector_all_mode_preserves_complete_prompt_jobs(self):
        image_batch = torch.rand((3, 4, 4, 3))
        references = [
            json.dumps({"tile_index": index, "tile_id": f"T{index + 1:03d}"})
            for index in range(3)
        ]
        selected = SmartSamplerTileSelector().select(
            [image_batch],
            ["p1", "p2", "p3"],
            ["n1", "n2", "n3"],
            references,
            [11, 12, 13],
            ["All tiles (production)"],
            [2],
        )

        self.assertEqual([len(values) for values in selected], [3, 3, 3, 3, 3])
        self.assertEqual(selected[1], ["p1", "p2", "p3"])
        self.assertEqual(selected[4], [11, 12, 13])

    def test_sampler_tile_selector_rejects_unaligned_or_missing_jobs(self):
        selector = SmartSamplerTileSelector()
        references = [
            json.dumps({"tile_index": 0}),
            json.dumps({"tile_index": 1}),
        ]
        with self.assertRaisesRegex(ValueError, "not aligned"):
            selector.select(
                [torch.rand((2, 4, 4, 3))],
                ["p1"],
                ["n1", "n2"],
                references,
                [1, 2],
                ["One tile test"],
                [1],
            )
        with self.assertRaisesRegex(ValueError, "T003 is not available"):
            selector.select(
                [torch.rand((2, 4, 4, 3))],
                ["p1", "p2"],
                ["n1", "n2"],
                references,
                [1, 2],
                ["One tile test"],
                [3],
            )

    def test_sampler_test_tile_is_restored_at_its_original_stitch_position(self):
        source_tiles = torch.zeros((3, 4, 4, 3), dtype=torch.float32)
        metadata = {
            "tile_width": 4,
            "tile_height": 4,
            "output_tile_width": 4,
            "output_tile_height": 4,
            "scale_factor": 1.0,
            "tiles": [
                {"tile_index": index, "source_index": 0}
                for index in range(3)
            ],
        }
        references = [
            json.dumps({"tile_index": index, "tile_id": f"T{index + 1:03d}"})
            for index in range(3)
        ]
        selected = SmartSamplerTileSelector().select(
            [source_tiles],
            ["p1", "p2", "p3"],
            ["n1", "n2", "n3"],
            references,
            [1, 2, 3],
            ["One tile test"],
            [2],
        )
        processed_tile = torch.ones_like(selected[0][0])
        merged = SmartTileMergePartialBatch().merge(
            [processed_tile],
            selected[3],
            [source_tiles],
            [json.dumps(metadata)],
            ["bilinear"],
        )[0]

        self.assertTrue(torch.equal(merged[1], torch.ones_like(merged[1])))
        self.assertTrue(torch.equal(merged[0], torch.zeros_like(merged[0])))
        self.assertTrue(torch.equal(merged[2], torch.zeros_like(merged[2])))

    def test_prompt_audit_displays_and_saves_the_complete_text_chain(self):
        references = [
            json.dumps({"tile_id": "T001", "tile_index": 0, "position": "top-left"}),
            json.dumps({"tile_id": "T002", "tile_index": 1, "position": "top-right"}),
        ]
        audits = [
            json.dumps(
                {
                    "raw_source_response": "white brick wall with narrow windows",
                    "local_source_evidence": "white brick wall",
                    "evidence_guard": "exact_structured_source",
                }
            ),
            json.dumps(
                {
                    "raw_source_response": "dark water crossed by subtle wakes",
                    "local_source_evidence": "dark water",
                    "evidence_guard": "exact_structured_source",
                }
            ),
        ]
        system = {
            "task_preset": "Time of Day",
            "workflow_instruction": "Apply the requested time of day.",
            "user_request": "Make this a realistic night image.",
            "user_instruction": "Apply the requested time of day. Make this a realistic night image.",
            "caption_detail": "Detailed",
            "prompt_strategy": "task_directed",
        }
        workflow_prompt = {
            "1": {"class_type": "LoadImage", "inputs": {"image": "source.png"}}
        }

        with tempfile.TemporaryDirectory() as temporary:
            with patch("nodes.audit._log_directory", return_value=Path(temporary)):
                output = SmartTilePromptAuditLog().save(
                    ["inspect tile one", "inspect tile two"],
                    ["night edit; white brick wall", "night edit; dark water"],
                    ["no new objects", "no new objects"],
                    audits,
                    references,
                    ["miss", "hit"],
                    [system],
                    ["inspect the complete image"],
                    ["waterfront city with continuous lighting"],
                    ["user and master instructions combined"],
                    ["visual-review"],
                    workflow_prompt=[workflow_prompt],
                )

            readable, json_path, text_path = output["result"]
            payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
            self.assertTrue(Path(text_path).is_file())
            self.assertTrue((Path(temporary) / "tile_prompt_audit_latest.json").is_file())
            self.assertTrue((Path(temporary) / "tile_prompt_audit_latest.txt").is_file())
            self.assertEqual(payload["tile_count"], 2)
            self.assertEqual(payload["source_inputs"][0]["image"], "source.png")
            self.assertEqual(payload["tiles"][0]["tile_id"], "T001")
            self.assertNotIn("white brick wall with narrow windows", readable)
            self.assertNotIn("dark water crossed by subtle wakes", readable)
            self.assertEqual(
                payload["tiles"][0]["raw_vlm_response"],
                "white brick wall with narrow windows",
            )
            self.assertEqual(
                payload["tiles"][1]["raw_vlm_response"],
                "dark water crossed by subtle wakes",
            )
            self.assertIn("night edit; white brick wall", readable)
            self.assertIn("night edit; dark water", readable)
            self.assertIn("Make this a realistic night image", readable)
            self.assertEqual(output["ui"]["prompt_audit_text"], [readable])

    def test_prompt_audit_does_not_force_global_vlm_in_direct_user_mode(self):
        audit = SmartTilePromptAuditLog()
        direct = [{"prompt_strategy": "direct_user"}]
        task_directed = [{"prompt_strategy": "task_directed"}]
        common = (["instruction"], ["positive"], ["negative"], ["{}"], ["{}"], ["hit"])

        self.assertEqual(
            audit.check_lazy_status(*common, direct, ["global instruction"], None, ["combined"], ["log"]),
            [],
        )
        self.assertEqual(
            audit.check_lazy_status(
                *common,
                task_directed,
                ["global instruction"],
                (None,),
                ["combined"],
                ["log"],
            ),
            ["global_context"],
        )


if __name__ == "__main__":
    unittest.main()
