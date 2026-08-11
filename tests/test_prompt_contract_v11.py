"""Focused regression tests for the v11 prompt contract.

These cover the four properties the v11 redesign is responsible for and that must not
regress:

  A. No discrete-object description leaks into a uniform (smooth) tile.
  B. Uniform / sparse tiles color-lock to one canonical surface phrase even when several
     surface candidates overlap the tile (kills cross-tile color drift).
  C. A discrete object spanning several tiles reuses one identical canonical phrase, but
     only in tiles that confirmed it from their own pixels.
  D. Known False Detections are scrubbed from every tile field, including the new object
     fields, and can never become conditioning.
  E. The master brief accepts a well-formed object_map, treats it as optional, and
     rejects a malformed one.
  F. The v11 cache tags are in force (old tags cannot silently be reused) and no
     test-scene name is hardcoded in production prompt text.

Pure-logic tests: they exercise the deterministic resolver / selectors / validators and
do not need a GPU or a running ComfyUI. They import the same modules the pipeline uses.
"""

import json
import re
import unittest
from pathlib import Path

from nodes.prompting import (
    SmartTilePromptResolver,
    _apply_known_false_detection_guard,
    _loads_tolerant,
    _merge_canonical_clause,
    _repair_artifact_language,
    _select_canonical_object,
    _select_canonical_surface,
)
from nodes.cache import _global_caption_problem
from nodes.universal_prompting import (
    UNIFIED_TASK_INSTRUCTIONS,
    _compact_global_context,
    _object_map_context,
    _surface_map_context,
    _unified_context_awareness,
)


def _surface(sid, identity, target, locations):
    return {
        "id": sid,
        "identity": identity,
        "source_appearance": f"source appearance of {identity}",
        "target_prompt": target,
        "matching_locations": locations,
    }


def _task_directed_system(**overrides):
    system = {
        "composition_mode": "task_directed",
        "prompt_strategy": "task_directed",
        "operation_mode": "faithful_upscale",
        "task_preset": "Upscale / Detailer",
        "edit_action": "Upscale this image with realistic detail",
        "user_instruction": "",
        "direct_prompt": "Upscale this image with realistic detail",
        "known_false_detections": (),
        "prompt_format": "instruction_edit",
    }
    system.update(overrides)
    return system


def _resolve(reference, response, system=None, negative="artifacts, seams"):
    system = system or _task_directed_system()
    resolver = SmartTilePromptResolver()
    positive, neg, audit, _ref = resolver.resolve(
        json.dumps(response), json.dumps(reference), system, negative
    )
    return positive, neg, json.loads(audit)


class NoObjectLeakIntoUniformTile(unittest.TestCase):
    def test_uniform_tile_never_emits_a_discrete_object(self):
        # A uniform water tile that is offered an object candidate and even echoes it
        # must still resolve to the canonical surface only.
        reference = {
            "tile_index": 3,
            "evidence_class": "uniform",
            "visual_complexity": "simple",
            "canonical_surfaces": [
                _surface("water_main", "the main water body", "calm blue water", ["bottom left"])
            ],
            "canonical_objects": [
                _surface("pier_1", "a long pier", "a long wooden pier", ["bottom left"])
            ],
        }
        response = {
            "local_caption": "smooth calm water surface filling the tile",
            "dominant_region": "smooth calm water surface",
            "visible_boundaries": "",
            "surface_id": "water_main",
            "surface_prompt": "calm blue water",
            "object_id": "",
            "object_prompt": "",
            "local_features": "",
            "corrections_applied": "",
            "target_prompt": "smooth calm water surface",
        }
        positive, _neg, audit = _resolve(reference, response)
        self.assertNotIn("pier", positive.lower())
        self.assertEqual(audit["canonical_object_prompt"], "")
        self.assertEqual(positive, "Upscale this image with realistic detail. calm blue water.")


class UniformTileNoSceneBleed(unittest.TestCase):
    def _master(self):
        return {
            "scene_type": "aerial urban waterfront",
            "geographic_context": "likely Toronto",
            "view": "high-angle aerial",
            # The surface map licenses the "waterfront" claim in the scene label.
            "surface_map": [
                {
                    "id": "open_water",
                    "locations": ["bottom left"],
                    "identity": "open water",
                    "target_prompt": "calm blue open water",
                }
            ],
            "object_map": [],
        }

    def test_uniform_tile_gets_camera_only_no_scene_or_place(self):
        ctx = _compact_global_context(self._master(), "uniform")
        self.assertNotIn("urban waterfront", ctx)
        self.assertNotIn("Toronto", ctx)
        self.assertIn("aerial", ctx.lower())

    def test_structured_tile_keeps_scene_context(self):
        ctx = _compact_global_context(self._master(), "structured")
        self.assertIn("urban waterfront", ctx)

    def test_unlicensed_water_claim_is_stripped_for_structured_tiles(self):
        master = self._master()
        master["surface_map"] = []
        ctx = _compact_global_context(master, "structured")
        self.assertNotIn("waterfront", ctx)
        self.assertIn("aerial urban", ctx)


class StructuredTileKeepsOwnDetail(unittest.TestCase):
    def test_building_tile_is_not_overwritten_by_a_ground_surface(self):
        # A structured tile that overlaps an over-broad "ground" surface must keep its own
        # building/material description and NOT be replaced by the surface phrase.
        reference = {
            "tile_index": 0,
            "evidence_class": "structured",
            "visual_complexity": "complex",
            "canonical_surfaces": [
                _surface("ground", "urban ground surface", "urban ground surface with asphalt and grass",
                         ["top left", "top center", "top right", "middle left", "center", "bottom left"])
            ],
        }
        response = {
            "local_caption": "glass and concrete high-rise towers with many windows",
            "dominant_region": "high-rise towers",
            "visible_boundaries": "",
            "surface_id": "", "surface_prompt": "", "object_id": "", "object_prompt": "",
            "local_features": "",
            "corrections_applied": "",
            "target_prompt": "cluster of glass and concrete high-rise towers, reflective windows, "
                             "rooftop mechanicals, at center and right",
        }
        positive, _neg, audit = _resolve(reference, response)
        self.assertIn("high-rise towers", positive.lower())
        self.assertIn("glass and concrete", positive.lower())
        self.assertNotIn("urban ground surface", positive.lower())
        self.assertEqual(audit["canonical_surface_prompt"], "")

    def test_structured_tile_still_reuses_a_confirmed_spanning_object(self):
        obj = _surface("bridge_1", "a steel bridge", "a long grey steel bridge", ["center", "middle right"])
        reference = {"tile_index": 4, "evidence_class": "structured", "visual_complexity": "complex",
                     "canonical_objects": [obj]}
        response = {
            "local_caption": "steel bridge deck with buildings behind",
            "dominant_region": "steel bridge",
            "visible_boundaries": "", "surface_id": "", "surface_prompt": "",
            "object_id": "bridge_1", "object_prompt": "a long grey steel bridge",
            "local_features": "office buildings behind",
            "corrections_applied": "",
            "target_prompt": "a grey steel bridge deck with glass office buildings behind at upper right",
        }
        positive, _neg, _audit = _resolve(reference, response)
        self.assertIn("office buildings", positive.lower())
        self.assertIn("a long grey steel bridge", positive)


class UniformColorLock(unittest.TestCase):
    def test_multiple_surface_candidates_still_lock_to_one_phrase(self):
        # Two overlapping surface candidates; the tile gives no surface_id and generic
        # wording with no identity-token overlap. The old code returned None here and let
        # the model wording drift; the hardened selector must lock to one candidate.
        candidates = [
            _surface("primary", "the northern reservoir", "deep teal water", ["bottom left"]),
            _surface("secondary", "the southern lagoon", "deep teal water", ["bottom left"]),
        ]
        payload = {
            "local_caption": "a smooth continuous liquid expanse",
            "dominant_region": "a smooth continuous liquid expanse",
            "surface_prompt": "",
            "surface_id": "",
        }
        chosen = _select_canonical_surface({"canonical_surfaces": candidates}, payload, "uniform")
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["id"], "primary")  # deterministic: most location-specific first

    def test_two_neighbor_uniform_tiles_get_identical_target(self):
        candidates = [
            _surface("primary", "the northern reservoir", "deep teal water", ["bottom left"]),
            _surface("secondary", "the southern lagoon", "pale green water", ["bottom left"]),
        ]
        response = {
            "local_caption": "a smooth continuous liquid expanse",
            "dominant_region": "a smooth continuous liquid expanse",
            "visible_boundaries": "",
            "surface_id": "",
            "surface_prompt": "",
            "object_id": "",
            "object_prompt": "",
            "local_features": "",
            "corrections_applied": "",
            "target_prompt": "a smooth continuous liquid expanse",
        }
        left = {"tile_index": 1, "evidence_class": "uniform", "visual_complexity": "simple",
                "canonical_surfaces": candidates}
        right = {"tile_index": 2, "evidence_class": "uniform", "visual_complexity": "simple",
                 "canonical_surfaces": candidates}
        pos_left, _n1, _a1 = _resolve(left, response)
        pos_right, _n2, _a2 = _resolve(right, response)
        self.assertEqual(pos_left, pos_right)
        self.assertIn("deep teal water", pos_left)


class SpanningObjectIdentity(unittest.TestCase):
    def test_confirmed_object_reuses_one_verbatim_phrase_across_tiles(self):
        obj = _surface("bridge_1", "a steel bridge", "a long grey steel bridge with lattice trusses",
                       ["center", "middle right"])
        base_response = {
            "local_caption": "part of a steel bridge span crossing the frame",
            "dominant_region": "steel bridge deck",
            "visible_boundaries": "bridge continues past the right edge",
            "surface_id": "",
            "surface_prompt": "",
            "object_id": "bridge_1",
            "object_prompt": "a long grey steel bridge with lattice trusses",
            "local_features": "riveted railing",
            "corrections_applied": "",
            "target_prompt": "a sharp steel bridge span",
        }
        t_center = {"tile_index": 4, "evidence_class": "structured", "visual_complexity": "complex",
                    "canonical_objects": [obj]}
        t_right = {"tile_index": 5, "evidence_class": "structured", "visual_complexity": "complex",
                   "canonical_objects": [obj]}
        pos_c, _n, audit_c = _resolve(t_center, base_response)
        pos_r, _n2, audit_r = _resolve(t_right, base_response)
        phrase = "a long grey steel bridge with lattice trusses"
        self.assertIn(phrase, pos_c)
        self.assertIn(phrase, pos_r)
        self.assertEqual(audit_c["canonical_object_prompt"], audit_r["canonical_object_prompt"])

    def test_object_not_confirmed_is_never_drawn(self):
        # Candidate present by location, but the tile does not return the id and its own
        # evidence never names the object -> object must not appear (Problem A stays safe).
        obj = _surface("bridge_1", "a steel bridge", "a long grey steel bridge", ["center"])
        response = {
            "local_caption": "open sky with faint cloud texture",
            "dominant_region": "open sky",
            "visible_boundaries": "",
            "surface_id": "",
            "surface_prompt": "",
            "object_id": "",
            "object_prompt": "",
            "local_features": "",
            "corrections_applied": "",
            "target_prompt": "clear open sky",
        }
        selected = _select_canonical_object({"canonical_objects": [obj]}, response)
        self.assertIsNone(selected)
        reference = {"tile_index": 6, "evidence_class": "structured", "visual_complexity": "moderate",
                     "canonical_objects": [obj]}
        positive, _neg, _audit = _resolve(reference, response)
        self.assertNotIn("bridge", positive.lower())


class KnownFalseDetectionOverride(unittest.TestCase):
    def test_false_term_scrubbed_from_object_fields(self):
        response = json.dumps({
            "local_caption": "open water near the shore",
            "dominant_region": "open water",
            "visible_boundaries": "",
            "surface_id": "",
            "surface_prompt": "",
            "object_id": "pier_1",
            "object_prompt": "a long stone pier",
            "local_features": "a pier extending out",
            "corrections_applied": "",
            "target_prompt": "clean open water",
        })
        cleaned = json.loads(_apply_known_false_detection_guard(response, ["pier"]))
        blob = " ".join(str(v) for v in cleaned.values()).lower()
        self.assertNotIn("pier", blob)


class MasterObjectMapValidation(unittest.TestCase):
    def _brief(self, **extra):
        brief = {
            "image_description": (
                "a wide daytime aerial view of a mixed urban waterfront with low buildings "
                "roads open water and scattered greenery under even light"
            ),
            "global_appearance": "a clear balanced natural daytime daylight photograph",
            "regional_map": ["bottom left: open water"],
            "identity_anchors": [],
            "surface_map": [
                {"id": "w1", "locations": ["bottom left"], "identity": "the harbour water",
                 "source_appearance": "grey-blue water", "target_prompt": "calm grey-blue harbour water"}
            ],
            "material_map": [],
        }
        brief.update(extra)
        return json.dumps(brief)

    def _system(self):
        return {"unified_instruction_ui": True, "known_false_detections": ()}

    def test_valid_object_map_passes(self):
        brief = self._brief(object_map=[
            {"id": "b1", "locations": ["center"], "identity": "a bridge",
             "source_appearance": "steel bridge", "target_prompt": "a long steel bridge"}
        ])
        self.assertEqual(_global_caption_problem(brief, self._system()), "")

    def test_missing_object_map_is_allowed(self):
        self.assertEqual(_global_caption_problem(self._brief(), self._system()), "")

    def test_malformed_object_map_is_rejected(self):
        brief = self._brief(object_map="not a list")
        self.assertIn("object_map", _global_caption_problem(brief, self._system()))


class CanonicalClauseMerge(unittest.TestCase):
    CANONICAL = (
        "glass-clad high-rise buildings with reflective surfaces, sharp edges, and "
        "clean lines; marina with white boats docked at concrete piers"
    )

    def test_a_clause_the_tile_already_said_is_not_repeated(self):
        # Real T002: the whole canonical phrase was appended because the
        # whole-string containment test failed, so the buildings clause landed
        # in the prompt twice.
        lead = (
            "Glass-clad high-rise buildings with reflective surfaces, sharp edges, "
            "and clean lines; concrete structures with rooftop features"
        )
        merged = _merge_canonical_clause(lead, self.CANONICAL)
        self.assertEqual(merged.lower().count("glass-clad high-rise buildings"), 1)
        self.assertIn("marina with white boats", merged)
        self.assertIn("rooftop features", merged)

    def test_unrelated_canonical_wording_is_appended_whole(self):
        merged = _merge_canonical_clause("Dark teal water with gentle ripples", self.CANONICAL)
        self.assertTrue(merged.startswith("Dark teal water"))
        self.assertIn("marina with white boats", merged)

    def test_an_empty_lead_leaves_the_canonical_phrase_alone(self):
        self.assertEqual(_merge_canonical_clause("", self.CANONICAL), self.CANONICAL)
        self.assertEqual(_merge_canonical_clause("a lone tile", ""), "a lone tile")


class ArtifactRepairLanguage(unittest.TestCase):
    def test_debris_and_damage_words_are_stripped(self):
        out = _repair_artifact_language(
            "a multi-story building and a smaller building, with some debris and parked vehicles"
        )
        self.assertNotIn("debris", out.lower())
        self.assertIn("parked vehicles", out.lower())
        self.assertIn("building", out.lower())

    def test_user_request_can_keep_a_damage_word(self):
        out = _repair_artifact_language("a broken window", keep_terms=("show a broken window",))
        self.assertIn("broken", out.lower())

    def test_repair_task_drops_how_the_capture_looks(self):
        # Real T014, 2026-07-30: this reached the sampler and asked Klein to
        # render low-resolution pixelation as if it were content.
        caption = (
            "A dense urban area with buildings, trees, and streets. Trees have "
            "irregular, leafy crowns. Streets are narrow and winding. The image "
            "has a low-resolution, pixelated quality"
        )
        out = _repair_artifact_language(caption, aggressive=True)
        self.assertNotIn("pixelated", out.lower())
        self.assertNotIn("low-resolution", out.lower())
        self.assertIn("leafy crowns", out.lower())
        self.assertIn("narrow", out.lower())

    def test_faithful_upscale_keeps_softness_as_real_content(self):
        # "Upscale / Detailer" tells the model that blur is part of the photo,
        # so the capture-quality sweep must not fire outside repair tasks.
        caption = "Out-of-focus background buildings, blurry distant signage, sharp foreground railings"
        out = _repair_artifact_language(caption, aggressive=False)
        self.assertIn("blurry", out.lower())
        self.assertIn("out-of-focus", out.lower())

    def test_user_request_can_keep_a_capture_quality_word(self):
        out = _repair_artifact_language(
            "a pixelated retro arcade screen",
            keep_terms=("keep the pixelated screen",),
            aggressive=True,
        )
        self.assertIn("pixelated", out.lower())

    def test_resolver_strips_debris_from_final_positive(self):
        reference = {"tile_index": 7, "evidence_class": "structured", "visual_complexity": "moderate"}
        response = {
            "local_caption": "two buildings with a street between them and some debris",
            "dominant_region": "urban buildings",
            "visible_boundaries": "",
            "surface_id": "", "surface_prompt": "", "object_id": "", "object_prompt": "",
            "local_features": "a street with parked vehicles and scattered debris",
            "corrections_applied": "",
            "target_prompt": "two multi-story buildings with a street and some debris",
        }
        positive, _neg, _audit = _resolve(reference, response)
        self.assertNotIn("debris", positive.lower())


class ReadableInstructions(unittest.TestCase):
    def test_default_instructions_are_plain_language(self):
        for name, text in UNIFIED_TASK_INSTRUCTIONS.items():
            # No internal jargon in the user-facing editable text.
            for jargon in ("MASTER SCENE PASS", "EXACT-TILE PASS", "canonical", "verbatim", "regional_map"):
                self.assertNotIn(jargon, text, f"{jargon} leaked into {name} instructions")
            self.assertIn("TASK:", text)

    def test_default_context_no_longer_stamps_recognized_names(self):
        # Dropping the "Verified names" line reverts to the safer clarify-only default,
        # which stops a guessed city name being stamped onto every tile.
        text = UNIFIED_TASK_INSTRUCTIONS["Google Image Enhance"]
        self.assertEqual(
            _unified_context_awareness(text), "Scene Type + Continuity (recommended)"
        )


class TolerantJson(unittest.TestCase):
    def test_clean_json_parses(self):
        obj = _loads_tolerant('{"a": 1, "b": ["x", "y"]}')
        self.assertEqual(obj["b"], ["x", "y"])

    def test_prose_wrapped_and_fenced_json_parses(self):
        obj = _loads_tolerant('Here is the brief:\n```json\n{"a": 1}\n```\nthanks')
        self.assertEqual(obj, {"a": 1})

    def test_truncated_json_is_recovered_not_rejected(self):
        # Simulates the 896-token cutoff: object cut off mid-value. The completed part
        # must survive instead of aborting the run.
        truncated = (
            '{"source_medium": "satellite", "surface_map": ['
            '{"id": "w1", "locations": ["center"], "identity": "the harbour water", '
            '"source_appearance": "grey-blue", "target_prompt": "calm grey-blue harbou'
        )
        obj = _loads_tolerant(truncated)
        self.assertIsNotNone(obj)
        self.assertEqual(obj["source_medium"], "satellite")

    def test_unrecoverable_text_returns_none(self):
        self.assertIsNone(_loads_tolerant("not json at all"))
        self.assertIsNone(_loads_tolerant(""))


class ContractHygiene(unittest.TestCase):
    def _module_sources(self):
        nodes_dir = Path(__file__).resolve().parents[1] / "nodes"
        return {p.name: p.read_text(encoding="utf-8") for p in nodes_dir.glob("*.py")}

    def test_current_cache_tags_in_force(self):
        cache_src = self._module_sources()["cache.py"]
        self.assertIn("qwen3vl_4b_fp8_master_scene_v22_uniform_surfaces_v13", cache_src)
        self.assertIn("qwen3vl_4b_fp8_local_target_v29_structured_detail_v13", cache_src)
        # Superseded contract defaults must not remain as node defaults.
        for stale in (
            "v19_canonical_surfaces",
            "v26_canonical_surfaces",
            "v20_canonical_v11",
            "v27_canonical_v11",
            "v21_canonical_v12",
            "v28_canonical_v12",
        ):
            self.assertNotIn(stale, cache_src)

    def test_no_hardcoded_test_scene_names_in_production_prompts(self):
        banned = re.compile(r"toronto|lake ontario|cn tower", re.IGNORECASE)
        for name, src in self._module_sources().items():
            self.assertIsNone(banned.search(src), f"hardcoded test-scene name found in {name}")


if __name__ == "__main__":
    unittest.main()
