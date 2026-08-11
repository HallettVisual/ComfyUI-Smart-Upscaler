"""Route every tile-aligned consumer through the Sampler Tile Test Selector.

Why this exists
---------------
The selector picks ONE complete sampler job (image, positive, negative,
reference, seed) after all tile prompts have been generated. If only some of
those paths go through it, ComfyUI pairs the one selected prompt with the full
tile list and renders the same tile over and over. Everything that must stay
aligned with the selected tile has to read from the selector:

  * the sampler's VAE encode (the starting latent) and any ControlNet hint
  * the tile prompt encoders and the seed
  * Color Match's ``source_tile``
  * the Finalizer's ``tile_references``
  * the Tile Inspector's images, prompts, and references

The All Prompts audit log stays wired upstream of the selector, so the full set
of tile prompts is still generated, cached, and logged in one-tile mode.

In "All tiles (production)" the selector is a transparent pass-through, so a
gated graph behaves exactly like an ungated one.

Usage:
    python scripts/gate_sampler_tile_selector.py <input.json> [output.json]
"""

import json
import sys
from pathlib import Path


# Selector output slot -> the consumer inputs that must follow the same tile.
GATED_INPUTS = {
    0: (  # tile_images
        ("SmartTileColorMatch", "source_tile"),
        ("SmartTileInspector", "source_images"),
        ("VAEEncode", "pixels"),
        ("QwenImageDiffsynthControlnet", "image"),
        ("ControlNetApplySD3", "image"),
    ),
    1: (("SmartTileInspector", "prompts"),),  # positive_prompts
    3: (  # tile_references
        ("SmartTileFinalizer", "tile_references"),
        ("SmartTileInspector", "tile_references"),
    ),
}

# Node types that only ever belong to one engine chain. Their inputs are gated
# only when they sit in the same group as the selector, so gating a Z-Turbo
# graph never reaches into a different engine's sampler.
ENGINE_LOCAL_TYPES = {"VAEEncode", "QwenImageDiffsynthControlnet", "ControlNetApplySD3"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def node_group(graph, node):
    """Title of the smallest titled group containing this node's position."""
    x, y = node["pos"][0], node["pos"][1]
    best, best_area = None, None
    for group in graph.get("groups", []):
        bounds = group.get("bounding")
        title = str(group.get("title", "")).strip()
        if not bounds or not title:
            continue
        gx, gy, gw, gh = bounds
        if not (gx <= x <= gx + gw and gy <= y <= gy + gh):
            continue
        area = gw * gh
        if best_area is None or area < best_area:
            best, best_area = title, area
    return best


def disconnect(graph, node, input_name):
    for item in node.get("inputs") or []:
        if item.get("name") != input_name:
            continue
        link_id = item.get("link")
        if link_id is None:
            return None
        item["link"] = None
        removed = None
        for index, link in enumerate(graph["links"]):
            if link[0] == link_id:
                removed = graph["links"].pop(index)
                break
        if removed:
            source = next(
                (n for n in graph["nodes"] if n["id"] == removed[1]), None
            )
            if source:
                slot = (source.get("outputs") or [])[removed[2]]
                slot["links"] = [
                    value for value in (slot.get("links") or []) if value != link_id
                ]
        return removed
    return None


def connect(graph, source_id, source_slot, target_node, input_name, link_type):
    target_input = next(
        item for item in target_node["inputs"] if item.get("name") == input_name
    )
    graph["last_link_id"] = int(graph["last_link_id"]) + 1
    link_id = graph["last_link_id"]
    graph["links"].append(
        [link_id, source_id, source_slot, target_node["id"], target_node["inputs"].index(target_input), link_type]
    )
    target_input["link"] = link_id
    source = next(n for n in graph["nodes"] if n["id"] == source_id)
    slot = source["outputs"][source_slot]
    slot["links"] = list(slot.get("links") or []) + [link_id]
    return link_id


def gate(graph):
    selectors = [n for n in graph["nodes"] if n["type"] == "SmartSamplerTileSelector"]
    if len(selectors) != 1:
        raise SystemExit(
            f"Expected exactly one Sampler Tile Test Selector, found {len(selectors)}."
        )
    selector = selectors[0]
    for item in selector.get("inputs") or []:
        if item.get("name") in (
            "tile_images",
            "positive_prompts",
            "negative_prompts",
            "tile_references",
            "tile_seeds",
        ) and item.get("link") is None:
            raise SystemExit(
                f"The selector's '{item['name']}' input is not connected; "
                "connect all five before gating."
            )
    selector_group = node_group(graph, selector)
    by_id = {n["id"]: n for n in graph["nodes"]}
    changes = []

    for slot, targets in GATED_INPUTS.items():
        link_type = selector["outputs"][slot].get("type", "*")
        for node_type, input_name in targets:
            for node in graph["nodes"]:
                if node["type"] != node_type:
                    continue
                if (
                    node_type in ENGINE_LOCAL_TYPES
                    and node_group(graph, node) != selector_group
                ):
                    continue
                target_input = next(
                    (
                        item
                        for item in (node.get("inputs") or [])
                        if item.get("name") == input_name
                    ),
                    None,
                )
                if target_input is None:
                    continue
                existing = next(
                    (
                        link
                        for link in graph["links"]
                        if link[0] == target_input.get("link")
                    ),
                    None,
                )
                if existing and existing[1] == selector["id"] and existing[2] == slot:
                    continue
                was = (
                    f"{by_id[existing[1]].get('title') or by_id[existing[1]]['type']}#{existing[1]}"
                    if existing
                    else "nothing"
                )
                disconnect(graph, node, input_name)
                connect(graph, selector["id"], slot, node, input_name, link_type)
                changes.append(
                    f"{node.get('title') or node['type']}#{node['id']}.{input_name}: "
                    f"{was} -> selector[{selector['outputs'][slot]['name']}]"
                )

    if selector.get("mode") != 0:
        selector["mode"] = 0
        changes.append("selector re-enabled (was bypassed)")
    return changes


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    source = Path(sys.argv[1])
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_name(
        f"{source.stem}-gated.json"
    )
    graph = load(source)
    changes = gate(graph)
    target.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(f"Wrote {target}")
    for change in changes:
        print(f"  {change}")
    if not changes:
        print("  (already gated)")


if __name__ == "__main__":
    main()
