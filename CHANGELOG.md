# Changelog

## Unreleased — 2026-09-24

**Faster prompting**

- The vision model no longer gets an empty "thinking" block appended to its
  prompt. Qwen3-VL Instruct was not trained on it: some tiles looped to the
  token limit (about 60 s each), failed, were asked again and fell back to a
  guessed prompt. Tiles now take 11-23 s and stop on their own.
- A guessed fallback prompt is no longer saved to the cache, so the next run
  asks again instead of reusing the guess.

**Reuse saved prompts**

- New Prompt Director setting *7. Reuse saved prompts (edited copy, same
  size)*. Off by default. On, a retouched copy of a picture already run reuses
  its scene summary and tile prompts; tiles that changed a lot are read fresh,
  and a different photo is never matched.

**Color**

- Color Match gains *original_colors* (the original's broad color at every
  pixel, the model's detail on top) and *automatic* (original colors, or no
  change when the task is Style Transfer or Time of Day - connect the Prompt
  Director's `prompt_system`). Automatic is the new default. On real Klein 9B
  tiles, neighbouring-tile color mismatch dropped from 5.2 to under 1 (0-255).
- The preset list is down to Automatic / Original colors / No color change /
  Manual. Saved graphs with a retired preset still load and run as before.

**Fewer false positives**

- A tile is no longer shown an object the scene summary places elsewhere,
  unless that object covers at least 40% of the frame. In logged runs a third
  of those hints were copied into the tile prompt.

## 1.2.0 — 2026-09-23

**Fewer settings**

- The two finishing nodes are now **one dropdown each**. Color Match shows its
  Quick Preset and ships on *Match source brightness*; Stitch Tiles shows its
  Quick Preset and ships on *Photo upscale, seams hidden*. The dials each preset
  writes moved under Advanced, and a fresh node's dials now hold exactly the
  values its shipped preset writes.
- `vision_model_id` is gone from both caption nodes. It named no model — it only
  namespaced the cache, which `cache_tag` already does. The value it contributed
  stays in the key, so captions cached before this still match.
- `max_length`, and the Job Director's `selection_mode` / `tile_number`, moved
  under Advanced. The Job Director's tile controls duplicate the Sampler Tile
  Test Selector, which is the one to use.
- Visible widgets across the eleven nodes: 34 → 21.

**Defaults**

- Tile Planner enlarges with **Lanczos** out of the box instead of an AI upscaler
  model, so a fresh graph needs no extra download and no extra wire.
- Tile Job Director selects **all tiles** and uses a **fixed** seed — one seed for
  every tile keeps texture consistent across seams.
- `divisible_by` moved under Advanced now that tile sizes are rounded for you.

**Tiles**

- Processed tile sizes are always a multiple of 32. Every sampler latent grid in
  use is 8, 16 or 32 pixels, and a model that rounds its own working size to 32
  (Qwen Image 2.1) now returns the tile at exactly the planned size. Settings
  already aligned to 32 — including `divisible_by 64` — keep their grid, tile
  count and overlap unchanged.

**Workflows**

- New release workflow: `workflow/Smart-Upscaler-Z-Turbo-v2.json`, replacing
  v1a. Ships at 2x, carries the new preset-first finishing settings, and renames
  the `Tiled_Image` wire to `All_Tiles` so it can no longer be confused with the
  selector's `Tiled_Images_Out`.
- The shipped master instruction no longer names example objects ("loose hair
  strands, wires, branches"). Tiles copy whatever their instruction shows them,
  and a wet-hair portrait once came back full of wires and branches.

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
