# ComfyUI Smart Upscaler

**An upscaler that reads your whole picture before it touches a single tile.**

Tiled upscalers work on one small square at a time, so they have to guess. A
crop of roof tiles becomes "brown texture". A patch of lake becomes "blue
gradient". Then the model paints that guess — and you get a picture that is
sharper but subtly wrong, with seams where neighbouring tiles guessed
differently.

Smart Upscaler does the obvious thing first. It looks at the **whole image
once**, works out what is actually in it, then writes a **separate, accurate
prompt for every tile** before anything is redrawn. Tile 7 knows it is showing
part of a slate roof. The four tiles covering the lake all get the *same*
wording for that water, so they cannot disagree at the seams.

Free and open source. MIT.

> Advanced workflows for other tasks and other models are on my website under
> **ComfyUI Workflows** — [hallett-ai.com](https://hallett-ai.com)

---

## Install

1. Copy this folder into `ComfyUI/custom_nodes/`:

   ```text
   ComfyUI/custom_nodes/ComfyUI-Smart-Upscaler
   ```

2. Install two companion packs from the ComfyUI Manager:
   **ComfyUI-KJNodes** and **rgthree-comfy**.

3. Download the models — see **[docs/MODELS.md](docs/MODELS.md)**. About 13 GB.

4. Copy `workflow/Upscale_Test_Image.png` into `ComfyUI/input/` — that is the
   sample picture the workflow opens with.

5. Restart ComfyUI.

6. Load `workflow/Smart-Upscaler-Z-Turbo-v1a.json`.

No Python dependencies beyond what ComfyUI already has.

---

## First run

The workflow is laid out as numbered boxes, left to right, with a note in each
one explaining what to change.

1. **Load your picture** — or leave the sample one in place.
2. **Leave the Prompt Director alone.** It already carries the instructions this
   workflow needs.
3. Set **scale_factor** in the Tile Planner. Start with 2.
4. Press Run.

When it finishes, box 6 has a before/after slider and the saved file. Box 3
shows the prompt that was written for every single tile.

**If part of the picture came out wrong, read that tile's prompt.** It is
almost always the reason, and it is almost always fixable from the Prompt
Director rather than by changing models.

---

## The two controls that matter

Everything else has a working default.

**Instructions** — the complete recipe. It sets what the job is, and its `TASK:`
line picks one of four built-in engines (Google Image Enhance, Style Transfer,
Time of Day, Upscale / Detailer). Edit it freely; it is plain English.

A **preset** is just this box plus the other Director fields saved together
under a name, so one click sets a whole self-consistent configuration. Presets
are grouped by model family in the dropdown, and that grouping matters more than
it sounds:

- **Edit models** (Klein, Qwen Edit) get an edit command plus a tile description
- **Denoise models** (Z-Image Turbo, SDXL, Flux) get a plain description only

Instruction models treat appended context as context; denoise models render
every noun you give them. Hand a denoise model an edit command and it paints
your command words into the picture.

**Prompt Detail Level** — how much verified local detail each tile prompt
carries. *Simple* is one short line. *Adaptive* is short for plain tiles and
rich for busy ones. *Complex* and *Detailed* add full materials, colours and
patterns. *Maximum* names every visible item, for samplers that only render what
is named. Lower is faster.

### Making your own presets

Set the Director fields the way you like, then click **Save My Preset to Disk**.
It appears in the dropdown under *Saved:* and survives updates.

To install a pack, drop its `.json` into `presets/packs/` and restart. Packs
load after the shipped file in filename order, so a pack can deliberately
replace an entry by reusing its name, and updating Smart Upscaler never
overwrites one. The format is documented in
[presets/packs/README.md](presets/packs/README.md).

---

## What is in the pack

Ten pipeline nodes plus one routing helper:

| Node | What it does |
|---|---|
| Prompt Director | The master prompt. Instructions, detail level, your request. |
| Output-Scale Tiles | Enlarges and plans the tile grid, with overlap and feather. |
| Whole-Image Analysis | Reads the picture once. Cached. |
| Tile Job Director | Cuts the exact tiles and hands each one its context. |
| Exact-Tile Prompt | Writes one prompt per tile. Cached. |
| Sampler Tile Test Selector | Optional. Render one tile to try a setting. |
| Color Match to Original | Optional. Pulls tiles back toward the source colours. |
| Stitch Tiles Into One Picture | Puts everything back together and hides the joins. |
| Tile Prompt Inspector | Click a tile, see its prompt. |
| All Prompts Viewer + Log | Every prompt, on screen and on disk. |
| Generator Switch | Only the selected model chain runs. |

The image model is an **isolated, replaceable block**. Anything that takes a
picture and a prompt and returns a picture can go in its place — the shipped
workflow uses Z-Image Turbo with a Tile ControlNet, but the prompting side does
not care.

The **Generator Switch** is what makes that practical. Build your chain, end it
with a `Set_` node, connect a matching `Get_` to a spare `generator_` input, and
pick it from the dropdown. Chains you do not select are never executed — no
model loads, no VRAM, no time — so one workflow can carry ten different models
at once and cost nothing for the nine it is not using.

Internal engines (prompt builder, resolver, blender, seed derivation) live in
`nodes/` but are deliberately not menu nodes.

---

## How it stays honest

Every prompt is built from what can actually be seen, and every automatic
correction **removes** unsupported claims — none of them invent content. Each
one has an escape hatch: the tile's own pixels, or words you typed yourself.

- Large flat areas — water, sky, open ground — are found by **measuring pixels**,
  not by asking a model. The whole-image pass must then identify each measured
  area, or it is asked again about that specific spot.
- A tile is never told about an object the whole-image pass placed somewhere
  else, unless the tile's own pixels confirm it.
- Prompts state what **should** be there, never the defect. "Cracked, smeared
  wall" becomes "clean rendered wall".
- A tile never repeats where its crop sits in the full image as if it were a
  position inside the crop.
- Nothing about the scene is hardcoded. It is all measured or read from your
  picture, and your **Not in this image** box is the final word.

You can see all of it. The prompt log records the instruction each tile was
given, what the model answered, every correction applied, and the final prompt.

---

## Reading the results

The **All Prompts Viewer + Log** node prints every tile prompt and writes two
files per run:

```text
ComfyUI/output/Smart-Upscaler/logs/tile_prompt_audit_latest.txt
ComfyUI/output/Smart-Upscaler/logs/tile_prompt_audit_latest.json
```

The `.txt` is the readable summary — scene, shared surfaces, main subject, then
one prompt per tile. The `.json` keeps everything: the exact instruction each
tile was given, the raw model response, which corrections fired, and the cache
state. That JSON is the tool for diagnosing a bad result without re-rendering
anything.

The final node is core ComfyUI `SaveImage`, saving PNG on purpose: ComfyUI
embeds the workflow in the file, so dragging a saved result back into ComfyUI
restores the whole graph and its settings. Converting to JPEG loses that.

---

## Speed and memory

Built and tested on **16 GB VRAM**. The pipeline is stage-ordered, so only one
large model is resident at a time — the vision model finishes every tile prompt
before the image model loads.

Everything is cached to disk: enlarged tiles, the whole-image read, and every
tile prompt. Change the scale and rerun, and the prompts are reused. Caches live
in your ComfyUI user directory under `smart_upscaler_cache/` and are safe to
delete at any time.

They cannot grow without bound. Enlarged tiles are capped at **2 GB**; prompts
are capped at **256 MB** and anything unused for **30 days** is deleted. Pruning
runs on every write, and a full or read-only disk reports `WRITE SKIPPED` rather
than throwing away a good result.

Levers, in order of effect:

1. **Prompt Detail Level** — *Simple* and *Adaptive* cost a fraction of *Maximum*
2. the **one-tile test** — try a setting on one tile instead of ninety
3. `cache_mode` on the tile prompt node — `bypass` skips image analysis entirely
4. `caption_max_side` on the tile prompt node
5. `upscale_batch_size` — lower this first if the enlarger itself runs out of VRAM

Keep sampler tiles at or below the shipped 1536-pixel maximum on 16 GB.

---

## Seams

If you can see the joins, raise **overlap** and **feather** in the Tile Planner.
On most images going from 32/16 to 128/64 costs **no extra tiles** — the planner
simply makes each tile slightly larger. Check the preflight box before and after.

If one tile is visibly a different shade from its neighbours, set the Finalizer's
`consistency_mode` to **Smooth gradient**: it corrects the drifting tile without
dragging correct neighbours with it.

---

## Documentation

- **[docs/MODELS.md](docs/MODELS.md)** — every model, with links and folders
- **[docs/ARCHITECTURE_AND_PERFORMANCE.md](docs/ARCHITECTURE_AND_PERFORMANCE.md)** — how it works inside

## Workflows

`workflow/Smart-Upscaler-Z-Turbo-v1a.json` is **the release workflow**, and the
only one in the repo. Z-Image Turbo with a Tile ControlNet, laid out as numbered
boxes in run order with a guidance note in each. It runs as downloaded, once
`Upscale_Test_Image.png` is in your `ComfyUI/input/` folder.

More advanced workflows — several generators pre-wired into one graph, tuned for
different tasks — are on my website under **ComfyUI Workflows**:
[hallett-ai.com](https://hallett-ai.com)

The optional ESRGAN enlarger in box 1 ships **unwired** on purpose. Out of the
box the planner uses Lanczos, which needs no download — and a loader pointing at
a model you do not have would stop the whole run before it started. To use one:
download a 4x model, pick it in that loader, drag its output into the Tile
Planner's `upscale_model` input, and set the planner's first dropdown to *AI
upscaler model*.

The last box is an empty generator template — a KSampler, a VAE pair and a text
encode, already fed with the tiles, prompts and per-tile seeds. Wire your own
model into it, then connect `Get_Your Own Generator` to a spare input on the
Generator Switch. It ships unconnected, so it costs nothing on a normal run.

## Tests

Run from the pack folder:

```text
python -m unittest discover -s tests
```

A few tests check tuned preset content. They skip when no preset pack is
installed, which is the normal state of a fresh clone.

## Development

ComfyUI discovers nodes through `NODE_CLASS_MAPPINGS` and
`NODE_DISPLAY_NAME_MAPPINGS` in `__init__.py`. Read
`docs/ARCHITECTURE_AND_PERFORMANCE.md` before changing cache keys, tile
numbering, or output geometry. Two rules were learned the hard way: every guard
must subtract rather than inject, and the Prompt Director's widget order can
never be changed.

To add a node: create a class in `nodes/`, give it the ComfyUI fields
(`INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`), then import and register
it in `nodes/__init__.py`.

## Licence

MIT. Use it, fork it, build on it.

**Commercial use is permitted** — but that covers this node pack only. The
models you generate with carry their own licences, and some restrict commercial
work. Check the licence of every model you use before selling the output.
