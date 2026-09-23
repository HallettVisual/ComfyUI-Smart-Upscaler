"""Structural integrity of the shipped workflow files.

Smart-Upscaler-Z-Turbo-v2.json is the workflow shipped to users. These checks
catch silent JSON-graph breakage: dangling links, GET nodes with no matching
SET, and Smart node types that are not registered.
"""

import json
from pathlib import Path
import unittest

import nodes as node_registry

WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "workflow"
RELEASE_WORKFLOW = "Smart-Upscaler-Z-Turbo-v2.json"
WORKFLOW_FILES = (RELEASE_WORKFLOW,)


def _load(name):
    return json.loads((WORKFLOW_DIR / name).read_text(encoding="utf-8"))


class WorkflowIntegrityTests(unittest.TestCase):
    def test_all_links_connect_existing_nodes(self):
        for name in WORKFLOW_FILES:
            workflow = _load(name)
            node_ids = {node["id"] for node in workflow["nodes"]}
            for link in workflow.get("links", []):
                self.assertIn(link[1], node_ids, f"{name}: link {link[0]} source")
                self.assertIn(link[3], node_ids, f"{name}: link {link[0]} target")

    def test_every_get_node_has_a_matching_set_node(self):
        for name in WORKFLOW_FILES:
            workflow = _load(name)
            set_keys = {
                str(node["widgets_values"][0])
                for node in workflow["nodes"]
                if node["type"] == "SetNode"
            }
            for node in workflow["nodes"]:
                if node["type"] == "GetNode":
                    self.assertIn(
                        str(node["widgets_values"][0]),
                        set_keys,
                        f"{name}: GetNode {node['id']} has no matching SetNode",
                    )

    def test_all_smart_nodes_are_registered(self):
        for name in WORKFLOW_FILES:
            workflow = _load(name)
            smart_types = {
                node["type"]
                for node in workflow["nodes"]
                if str(node["type"]).startswith("Smart")
            }
            self.assertEqual(
                smart_types - set(node_registry.NODE_CLASS_MAPPINGS),
                set(),
                name,
            )

    def test_sampler_tile_selector_is_registered_as_an_optional_pipeline_node(self):
        self.assertIn("SmartSamplerTileSelector", node_registry.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            node_registry.NODE_DISPLAY_NAME_MAPPINGS["SmartSamplerTileSelector"],
            "Sampler Tile Test Selector (Optional)",
        )

    def test_group_titles_are_unique(self):
        for name in WORKFLOW_FILES:
            workflow = _load(name)
            titles = [group.get("title") for group in workflow.get("groups", [])]
            self.assertEqual(len(titles), len(set(titles)), f"{name}: {titles}")

    def test_release_workflow_is_ready_to_hand_out(self):
        """Everything a first-time user depends on, checked in one place."""
        workflow = _load(RELEASE_WORKFLOW)
        nodes = {node["id"]: node for node in workflow["nodes"]}

        # Nothing muted or bypassed: a downloaded workflow must run as-is.
        stalled = [
            node.get("title") or node["type"]
            for node in workflow["nodes"]
            if node.get("mode", 0) != 0
        ]
        self.assertEqual(stalled, [], "release workflow ships muted/bypassed nodes")

        director = next(
            node
            for node in workflow["nodes"]
            if node["type"] == "SmartUnifiedPromptGuidance"
        )
        values = director["widgets_values"]
        # The only engine here is a denoise engine, so the prompt style must be
        # description - an edit command would be noise in its text encoder.
        self.assertEqual(values[5], "Plain description (SDXL, Flux, denoise)")
        # The colour repair is a rare fix and must ship switched off.
        self.assertTrue(str(values[6]).lower().startswith("off"))
        # Known False Detections is per-image and global while set. Shipping one
        # person's leftover bans silently deletes real content from everybody
        # else's pictures - a "wood" ban cost a wooden cabin its material.
        self.assertEqual(str(values[2]).strip(), "", "release ships leftover false detections")

        # The sampler must read the GATED prompt list, or a one-tile test pairs
        # every tile with the selected tile's prompt.
        selector = [
            node
            for node in workflow["nodes"]
            if node["type"] == "SmartSamplerTileSelector"
        ]
        self.assertEqual(len(selector), 1)
        gated = {
            str(node["widgets_values"][0])
            for node in workflow["nodes"]
            if node["type"] == "SetNode"
            and str(node["widgets_values"][0]).endswith("_Out")
        }
        # The "add your own engine" template ships a second, deliberately
        # unwired CLIPTextEncode. Pick the one in the live chain by its clip
        # input, not by whichever happens to come first in the node list.
        encoder = next(
            node
            for node in workflow["nodes"]
            if node["type"] == "CLIPTextEncode"
            and any(
                item["name"] == "clip" and item.get("link") is not None
                for item in node["inputs"]
            )
        )
        text_link = next(
            item for item in encoder["inputs"] if item["name"] == "text"
        )["link"]
        source = next(
            link for link in workflow["links"] if link[0] == text_link
        )[1]
        self.assertIn(
            str(nodes[source]["widgets_values"][0]),
            gated,
            "the sampler's prompt must come through the one-tile selector",
        )

        # Guidance the user reads before touching anything.
        markdown = [
            node for node in workflow["nodes"] if node["type"] == "MarkdownNote"
        ]
        self.assertGreaterEqual(len(markdown), 10)
        text = " ".join(str(node["widgets_values"][0]) for node in markdown)
        self.assertIn("krea2", text)
        self.assertIn("huggingface.co", text)
        for group in workflow["groups"]:
            self.assertRegex(str(group["title"]), r"^\d\.\s", "groups are numbered")

    def test_release_workflow_needs_no_file_the_user_was_not_told_to_download(self):
        """Optional extras must not gate the first run.

        ComfyUI validates every node reachable from an output node, and a
        loader whose file is missing fails the whole prompt with "Value not in
        list" before anything executes. A file that only exists on the author's
        machine therefore makes the workflow dead on arrival for everyone else -
        silently, because it runs perfectly here. Only the models the download
        note actually lists may sit in the live graph.
        """
        workflow = _load(RELEASE_WORKFLOW)
        nodes = {node["id"]: node for node in workflow["nodes"]}

        # The demo image must be one the downloader can actually obtain: either
        # ComfyUI's own example.png, or a file that ships in this repo.
        shipped = {item.name for item in WORKFLOW_DIR.iterdir() if item.is_file()}
        for node in workflow["nodes"]:
            if node["type"] == "LoadImage":
                name = str(node["widgets_values"][0])
                self.assertIn(
                    name,
                    shipped | {"example.png"},
                    "LoadImage points at a picture only the author has",
                )

        # The ESRGAN enlarger is optional and documented as needing no
        # download, so it must ship unwired rather than muted - a muted node
        # would trip the "runs as-is" check above.
        planner = next(
            node
            for node in workflow["nodes"]
            if node["type"] == "SmartUpscaledTilePlanner"
        )
        upscale_model = next(
            item for item in planner["inputs"] if item["name"] == "upscale_model"
        )
        self.assertIsNone(
            upscale_model["link"],
            "the optional AI upscaler is wired in, so a missing 4x model "
            "blocks the whole run",
        )

        # Nothing else may feed the graph from a loader we never documented.
        documented = {
            "qwen3vl_4b_fp8_scaled.safetensors",
            "qwen_3_4b.safetensors",
            "ae.safetensors",
            "z_image_turbo_nvfp4.safetensors",
        }
        reachable_loaders = {
            node["type"]: str(node["widgets_values"][0])
            for node in workflow["nodes"]
            if node["type"] in {"CLIPLoader", "VAELoader", "UNETLoader"}
        }
        for node_type, filename in reachable_loaders.items():
            self.assertIn(filename, documented, f"{node_type} loads an undocumented file")


if __name__ == "__main__":
    unittest.main()
