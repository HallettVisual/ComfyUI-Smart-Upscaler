# Changelog

## 1.1.0 — 2026-08-11

**Renamed**

- **Model Engine Switch → Generator Switch.** Its sockets are now
  `generator_1`…`generator_10` and `generator_number`, and the dropdown reads
  "Generator 1". "Engine" was doing two unrelated jobs — the prompt task and the
  model chain — so the model chain gave up the word. Graphs saved before this
  migrate themselves: old sockets are renamed on load with their wires intact,
  and old "Engine 3" dropdown values still select slot 3.
- `Finish and Blend Tiles` → **Stitch Tiles Into One Picture**.
- The advanced "Generation Engine (Automatic)" widget, which is about caption
  wording rather than image generation, is now **Caption Wording (Automatic)**.

**Fixed**

- The release workflow wired an optional ESRGAN loader into the Tile Planner.
  ComfyUI validates every node reachable from an output, so anyone without that
  exact 4x model had the whole run refused before a tile was drawn — despite the
  workflow's own note saying the enlarger needs no download. It now ships
  unwired, with instructions for connecting it.
- The workflow opened a picture that only existed on the author's machine, which
  failed the same way. It now ships with its sample image.

**Changed**

- **The Load Preset list ships empty.** The Prompt Director's defaults come from
  code, so nothing depends on it; the list fills from packs dropped into
  `presets/packs/` and from your own entries saved with *Save My Preset to Disk*.
- Removed the **Save Log With Image** option and its output. Logs still always
  land in `output/Smart-Upscaler/logs/`.

## 1.0.0 — 2026-07-30

First public release. The pack and the shipped workflow are frozen at this
point; everything below is what people are testing.

**The idea**

- Whole-image analysis runs once, and every tile prompt inherits it. A tile is
  never asked to guess what it is a crop of.
- Large flat surfaces — water, sky, open ground — are found by **measuring
  pixels**, not by asking a model. Every measured area must be identified by the
  whole-image pass, or it is asked again about that exact spot.
- Tiles showing the same surface get the same wording, so neighbouring tiles
  cannot disagree at a seam.
- Every automatic correction **subtracts** an unsupported claim. None of them
  inject content, and each has an escape hatch: the tile's own pixels, or words
  the user typed.

**Nodes**

Ten pipeline nodes plus the Model Engine Switch: Prompt Director, Output-Scale
Tiles, Whole-Image Analysis, Tile Job Director, Exact-Tile Prompt, Sampler Tile
Test Selector, Color Match to Original, Finish and Blend Tiles, Tile Prompt
Inspector, All Prompts Viewer + Log.

**Workflow**

- `Smart-Upscaler-Z-Turbo-v1.json` — Z-Image Turbo with a Tile ControlNet, six
  numbered boxes in run order.
- Box 7 is an optional empty engine template, shipped unconnected, for wiring in
  your own model.
- Ships with nothing muted or bypassed, no leftover false detections, and the
  colour repair switched off, so a fresh download runs as-is.

**Everything else**

- Preset pack, grouped by model family, that fills every Prompt Director control
  in one click.
- Output-scale adaptive tiling, overlap and feather, gradient seam correction.
- Persistent bounded caches for enlarged tiles, whole-image reads and tile
  prompts. A full or read-only disk reports `WRITE SKIPPED` rather than
  discarding a good result.
- Full prompt audit on screen and on disk — every instruction, every model
  answer, every correction that fired.
- One-tile test mode.
- 240 tests.

Built and tested on 16 GB VRAM. MIT.
