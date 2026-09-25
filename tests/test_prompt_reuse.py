import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from nodes.cache import SmartCachedTextGenerate, SmartCachedTilePromptGenerator
from nodes.universal_prompting import PROMPT_REUSE_CHOICES, SmartPromptGuidance


class CountingModel:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def tokenize(self, prompt, **kwargs):
        return {"tokens": [1]}

    def generate(self, tokens, **kwargs):
        self.calls += 1
        return [2]

    def decode(self, generated_ids):
        return self.text


TILE_CAPTION = json.dumps(
    {
        "local_caption": "red tiled roofs over pale stone walls",
        "dominant_region": "red tiled roofs",
        "visible_boundaries": "roof edges cross the tile",
        "corrections_applied": "",
        "target_prompt": "red tiled roofs over pale stone walls",
    }
)


def _picture(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((1, 48, 64, 3), generator=generator)


class PromptReuseTests(unittest.TestCase):
    def _tile_arguments(self, image, reuse, source_rect=(0, 0, 64, 48)):
        _, system, _ = SmartPromptGuidance().build(
            task_preset="Upscale / Detailer",
            caption_detail="Detailed",
            prompt_reuse=PROMPT_REUSE_CHOICES[1 if reuse else 0],
        )
        reference = json.dumps(
            {
                "tile_index": 0,
                "tile_id": "T001",
                "row": 0,
                "column": 0,
                "evidence_class": "structured",
                "source_rect": dict(zip(("x", "y", "width", "height"), source_rect)),
            }
        )
        return (
            image,
            "Caption exact tile T001.",
            reference,
            system,
            "Managed by Prompt Director (recommended)",
            "read_write",
            "reuse-test",
            "artifacts, seams",
        )

    def test_an_edited_copy_reuses_tile_prompts_without_the_model(self):
        original = _picture(1)
        edited = original.clone()
        edited[:, 10:20, 10:20, :] = 0.5  # a small local retouch
        different = _picture(2)
        model = CountingModel(TILE_CAPTION)
        node = SmartCachedTilePromptGenerator()
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch("nodes.cache._cache_root", return_value=Path(cache_directory)):
                first = node.generate(model, *self._tile_arguments(original, True))
                self.assertEqual(model.calls, 1)

                arguments = self._tile_arguments(edited, True)
                self.assertEqual(node.check_lazy_status(None, *arguments), [])
                reused = node.generate(None, *arguments)
                self.assertIn("REUSED", reused[4])
                self.assertEqual(reused[0], first[0])

                # Switched off, the edited copy is read fresh.
                self.assertEqual(
                    node.check_lazy_status(None, *self._tile_arguments(edited, False)),
                    ["clip"],
                )
                # A different photo is never matched, even with reuse on.
                self.assertEqual(
                    node.check_lazy_status(None, *self._tile_arguments(different, True)),
                    ["clip"],
                )
                # A finished result re-run at 1x: the same grid place and the
                # same look, but a larger crop at new input-pixel coordinates.
                finished = torch.nn.functional.interpolate(
                    edited.movedim(-1, 1), scale_factor=2, mode="bilinear"
                ).movedim(1, -1)
                rerun = self._tile_arguments(finished, True, (0, 0, 128, 96))
                self.assertEqual(node.check_lazy_status(None, *rerun), [])
                self.assertEqual(node.generate(None, *rerun)[0], first[0])
        self.assertEqual(model.calls, 1)

    def test_an_edited_copy_reuses_the_scene_summary(self):
        original = _picture(3)
        edited = (original + 0.01).clamp(0, 1)
        prompt, system, _ = SmartPromptGuidance().build(
            prompt_reuse=PROMPT_REUSE_CHOICES[1]
        )
        model = CountingModel('{"scene_type":"harbor town","recurring_details":[]}')
        node = SmartCachedTextGenerate()
        settings = (
            prompt,
            256,
            "Managed by Prompt Director (recommended)",
            False,
            True,
            "read_write",
            "reuse-brief-test",
        )
        with tempfile.TemporaryDirectory() as cache_directory:
            with patch("nodes.cache._cache_root", return_value=Path(cache_directory)):
                first = node.generate(model, original, *settings, prompt_system=system)
                self.assertEqual(
                    node.check_lazy_status(None, edited, *settings, prompt_system=system),
                    [],
                )
                reused = node.generate(None, edited, *settings, prompt_system=system)
        self.assertEqual(model.calls, 1)
        self.assertEqual(reused[0], first[0])
        self.assertIn("REUSED", reused[1])


if __name__ == "__main__":
    unittest.main()
