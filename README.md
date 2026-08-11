# ComfyUI Smart Upscaler

**Every tile gets its own prompt — written after reading the whole picture.**

A normal tiled upscaler sees a crop of roof and thinks *brown texture*. It sees a
patch of lake and thinks *blue gradient*. Then the model paints those guesses,
and you get something sharper but subtly wrong, with seams where neighbouring
tiles guessed differently.

Smart Upscaler looks at the **whole image first**. Tile 7 knows it is part of a
slate roof. The four tiles covering the lake all get the *same* wording for that
water, so they cannot disagree at the seam.

Free and open source. MIT.

<!-- Before/after comparison goes here. -->

---

## How it works

1. The whole image is read **once**, and the result is cached.
2. The image is enlarged and divided into overlapping tiles.
3. Every tile gets its own prompt, built from its own pixels **plus** the shared
   scene context.
4. Your image model redraws the tiles.
5. The tiles are stitched back together and the joins are hidden.
6. You can click any tile and see the exact prompt that made it.

Steps 1–3 and 6 are what this pack does. **Step 4 is yours** — anything that
takes an image and a prompt and returns an image can sit there.

---

## Install

1. Put this folder in `ComfyUI/custom_nodes/ComfyUI-Smart-Upscaler`
2. From ComfyUI Manager, install **ComfyUI-KJNodes** and **rgthree-comfy**
3. Download the models listed in **[docs/MODELS.md](docs/MODELS.md)** — about 13 GB
4. Copy `workflow/Upscale_Test_Image.png` into your `ComfyUI/input/` folder
5. Restart ComfyUI
6. Load `workflow/Smart-Upscaler-Z-Turbo-v1a.json`

No Python dependencies beyond ComfyUI itself.

> **Step 4 matters.** ComfyUI refuses to run a workflow that points at an image
> it cannot find, so do it before you open the graph — or just pick a picture of
> your own once it's loaded.

---

## First run

The workflow is laid out in numbered boxes, left to right, with a note in each.

1. Load your picture — or leave the sample one.
2. Set **scale_factor** in the Tile Planner. Start with `2`.
3. Leave the Prompt Director alone. It arrives already filled in.
4. Press **Run**.

**If part of the picture comes out wrong, read that tile's prompt.** It is almost
always the reason, and it is almost always fixable from the Prompt Director
rather than by changing models.

---

## The two controls that matter

Everything else has a working default.

### Instructions

Plain English, and the whole recipe. It says what the job is — detail
enhancement, restoration, style change, time of day. Edit it directly.

### Prompt Detail Level

How much verified local detail each tile prompt carries.

| Setting | What each tile prompt looks like |
|---|---|
| **Simple** | One short line |
| **Adaptive** | Short for plain tiles, rich for busy ones |
| **Complex** / **Detailed** | Full materials, colours and patterns |
| **Maximum** | Names every visible item |

More is not better — it depends on your model. Lower is faster.

---

## Presets

**The Load Preset list starts empty.** That is deliberate: the Prompt Director's
defaults come from code, and the shipped workflow carries its own tuned
instructions, so nothing depends on it.

It fills up two ways:

- Set the fields how you like, then click **Save My Preset to Disk**. Your entry
  appears under *Saved:* and survives updates.
- Drop a pack's `.json` into `presets/packs/` and restart. Format is documented
  in [presets/packs/README.md](presets/packs/README.md).

---

## Edit models vs denoise models

They need different prompt grammar, and getting it wrong is the most common way
to produce a bad result with no error message.

- **Edit models** (Klein, Qwen Edit) take an edit command **plus** a tile description.
- **Denoise models** (Z-Image Turbo, SDXL, Flux) take a plain description **only**.

Denoise models render every noun you give them. Hand one an edit command and it
paints your command words into the picture.

---

## Generator Switch

The image model is a separate, replaceable block — and you can wire in **up to
ten of them** in a single workflow.

Build a chain, end it with a `Set_` node, connect a matching `Get_` to a spare
`generator_` input, and pick it from the dropdown.

**Chains you do not select never execute.** No model loads, no VRAM, no time. One
workflow can carry Z-Turbo, Klein, SDXL and Flux at once and cost nothing for the
three it isn't using.

---

## What's in the pack

| Node | What it does |
|---|---|
| Prompt Director | The master prompt. Instructions, detail level, your request. |
| Output-Scale Tiles | Enlarges and plans the tile grid, with overlap and feather. |
| Whole-Image Analysis | Reads the picture once. Cached. |
| Tile Job Director | Cuts the exact tiles and hands each one its context. |
| Exact-Tile Prompt | Writes one prompt per tile. Cached. |
| Sampler Tile Test Selector | Optional. Render one tile to try a setting. |
| Color Match to Original | Optional. Pulls tiles back toward the source colours. |
| Stitch Tiles Into One Picture | Puts it back together and hides the joins. |
| Tile Prompt Inspector | Click a tile, see its prompt. |
| All Prompts Viewer + Log | Every prompt, on screen and on disk. |
| Generator Switch | Only the selected model chain runs. |

---

## How it stays honest

Every prompt is built from what can actually be seen, and every automatic
correction **removes** an unsupported claim — none of them invent content.

- Large flat areas — water, sky, open ground — are found by **measuring pixels**,
  not by asking a model. The whole-image pass must then identify each measured
  area, or it is asked again about that exact spot.
- A tile is never told about an object the whole-image pass placed somewhere
  else, **unless the tile's own pixels confirm it**.
- Prompts state what *should* be there, never the defect. "Cracked, smeared wall"
  becomes "clean rendered wall".
- Nothing about the scene is hardcoded. Your **Not in this image** box is the
  final word.

The point is not to make the model invent detail. It is to give it better context
for rebuilding detail that belongs there.

---

## Logs and diagnosis

Click through tiles with the **Tile Prompt Inspector**, or read the full record:

```text
ComfyUI/output/Smart-Upscaler/logs/
```

The `.txt` is the readable summary. The `.json` keeps everything — the exact
instruction each tile was given, the raw model response, which corrections fired,
and the cache state. **That JSON is how you diagnose a bad tile without
re-rendering anything.**

---

## Speed, memory and caching

Built and tested on **16 GB VRAM**. The pipeline is stage-ordered, so only one
large model is resident at a time — the vision model finishes every tile prompt
before the image model loads.

The whole-image read, the enlarged tiles and every tile prompt are cached, so
changing the scale and rerunning reuses the prompts. The cache lives in your
ComfyUI user directory under `smart_upscaler_cache/` and is safe to delete any
time.

**It cannot grow without bound:** enlarged tiles are capped at 2 GB, prompts at
256 MB, and anything unused for 30 days is deleted. Pruning runs on every write.

If you need it faster, in order of effect:

1. **Prompt Detail Level** — *Simple* and *Adaptive* cost a fraction of *Maximum*
2. **The one-tile test** — try a setting on one tile instead of ninety
3. `cache_mode` set to `bypass` — skips image analysis entirely
4. `upscale_batch_size` — lower this first if the enlarger runs out of VRAM

---

## Seams

Raise **overlap** and **feather** in the Tile Planner. Going from 32/16 to 128/64
usually costs **no extra tiles** — the planner just makes each tile slightly
larger. Check the preflight box before and after.

If one tile is visibly a different shade from its neighbours, set the stitcher's
`consistency_mode` to **Smooth gradient**. It corrects the drifting tile without
dragging correct neighbours with it.

---

## The included workflow

`workflow/Smart-Upscaler-Z-Turbo-v1a.json` — Z-Image Turbo with a Tile
ControlNet. It runs as downloaded.

The optional ESRGAN enlarger ships **unwired on purpose**. Out of the box the
planner uses Lanczos, which needs no download — and a loader pointing at a model
you don't have would stop the whole run before it started. To use one: download a
4x model, pick it in that loader, drag its output into the Tile Planner's
`upscale_model` input, and switch the planner's first dropdown to *AI upscaler
model*.

The last box is an empty generator template, already fed with tiles, prompts and
per-tile seeds. Wire your own model into it and connect it to the Generator
Switch. It ships unconnected, so it costs nothing.

---

## More workflows

Advanced Smart Upscaler workflows — several generators pre-wired into one graph,
tuned for different tasks — and other ComfyUI tools:

**[hallett-ai.com](https://hallett-ai.com)**

---

## Documentation

- **[docs/MODELS.md](docs/MODELS.md)** — every model, with links and folders
- **[docs/ARCHITECTURE_AND_PERFORMANCE.md](docs/ARCHITECTURE_AND_PERFORMANCE.md)** — contracts, caches, performance

## Licence

MIT. Use it, modify it, build on it.

**Commercial use of this node pack is permitted.** The AI models you generate
with have their own licences, and some restrict commercial work — check the
licence of every model you use before selling the output.
