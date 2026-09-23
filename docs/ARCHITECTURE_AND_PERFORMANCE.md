# Smart Upscaler Architecture and Performance Notes

Reference notes for anyone changing this pack: the contracts that must hold, how
the caches are keyed, and where the performance actually goes.

## Active Scope

The release workflow is `workflow/Smart-Upscaler-Z-Turbo-v2.json`. The goal is a
ComfyUI-native regional prompt upscaler — not a model loader and not a
replacement sampler stack. The image model block is an example and stays
externally replaceable: anything that takes a picture and a prompt and returns a
picture can go in its place.

## Stable Contracts

- Tile indexes are zero-based internally and displayed as `T001`, `T002`, and so on.
- Overlay labels, prompt references, selector values, sampler batches, partial merge, reviewer images, and prompt text must use the same tile reference.
- `SmartUpscaledTilePlanner` owns output-scale geometry, optional AI or standard-resize preprocessing, masks, metadata, the baseline image, and the diagnostic preview. Its `UPSCALE_MODEL` input is lazy and required only for the AI method.
- Model-specific diffusion remains external. Smart nodes should continue to exchange standard `IMAGE`, `MASK`, and `STRING` values.
- Optional source color correction remains a visible `SmartTileColorMatch` stage. Generated tiles then pass through `SmartTileFinalizer` for source-reference correction, overlap-only cross-tile color consistency, partial merge, and feather blending. Every source input must resolve to the matching enlarged tile.
- `SmartUnifiedPromptGuidance` is displayed as `Prompt Director: Editable Master Instructions`. The v10 workflow exposes exactly three text fields. TASK, PROMPT DETAIL, IMAGE ANALYSIS, and GLOBAL CONTEXT settings are readable lines inside Instructions instead of separate hidden or competing controls. Its third output is a visible runtime blueprint containing the exact full-image instruction, spatial filtering rules, tile contract, and final Klein prompt formula. Built-in instructions are deliberately compact; the runtime blueprint remains complete. A non-serialized UI dropdown lists built-in presets and user presets loaded from `presets/user_prompt_presets.json`; saving writes all three fields atomically to that disk file. Browser-local presets are migrated once when absent from disk. The earlier `SmartPromptGuidance` node remains registered so older workflows retain their original controls.
- The global VLM analyzes the clean fully enlarged composite once and returns a master scene prompt with rich image description, generic scene type, geography when supportable, view, source/artifact interpretation, spatial regional map, useful recurring details, and verified identity anchors. VLM-authored target prose and full-image object inventories are not passed to literal local edit models; the Prompt Director supplies the deterministic target task. Proper names are spatially filtered before reaching a tile and remain subject to exact local visual confirmation.
- Tile captioning receives only the exact unmodified output-scale crop. Whole-image identity and continuity arrive separately as compact text. Do not reintroduce a locator, mask, colored border, or second image panel: literal caption/edit models can describe those diagnostics as source content and reproduce them in the final image.
- The master `surface_map` assigns every large cross-tile surface explicit coarse locations and one canonical target phrase. Sparse and uniform tiles receive no regional object descriptions. A uniform or sparse tile always locks to a single canonical surface phrase when any candidate overlaps it (ranked by location specificity, then identity-token match), rather than falling through to the local model's alternate wording — this prevents cross-tile color drift on smooth water/sky/wall regions.
- The master `object_map` is the discrete-object twin of `surface_map`: one canonical phrase per large object that spans multiple tiles (bridge, tower, building face). A tile reuses the canonical `object_prompt` verbatim only when its own pixels confirm the object (via a returned `object_id` or an unambiguous identity-token match), so a spanning object looks identical everywhere without ever being drawn into a tile whose pixels do not show it. Uniform tiles are never offered object candidates.
- The local VLM returns `local_caption`, `dominant_region`, `visible_boundaries`, `corrections_applied`, and `target_prompt`. Only `target_prompt` enters positive conditioning. It must apply the task brief to exact locally visible content and describe the desired result, not preserve source defects as requested content. Negative prompts remain deterministic.
- Structured tiles receive source/geographic interpretation but no complete-image object inventory or VLM-authored target embellishment. The exact crop remains the authority for presence. The visible `Tile Detail Level` control writes Simple, Adaptive, Complex, or Detailed into the authoritative `PROMPT DETAIL` line. No scene, landmark, place, or object identity is hard-coded.
- Deterministic pixel evidence distinguishes uniform surfaces, sparse boundary content, and moderate/complex/dense structured crops. It is a prompt-detail and hallucination guard, not a semantic classifier.
- Known false detections are analysis constraints, never diffusion negatives. Both global and local cache workers reject matching whole terms or phrases and retry once with the original full instruction. If the local VLM repeats a confirmed false concept, deterministic code removes the affected caption clause and revalidates; sampling still stops if no safe, meaningful local or target caption remains.
- `Use User Request only (no image analysis)` maps to `direct_user` and must remain a true lazy bypass. Neither global nor tile VLM generation executes, and the positive prompt equals the User Request, or the preset's short default action when the request is blank.
- Model loaders, conditioning, scheduler, sampler, and decoder remain external and replaceable. Smart orchestration nodes must not assume Klein, Flux Dev, or SDXL internals.
- Overlap and feather settings are specified in final output pixels. Tile dimensions are rounded up to `divisible_by`, and then up to a multiple of 32: every sampler latent grid in use is 8, 16, or 32 pixels, and a model that snaps its own working size to 32 (Qwen Image 2.1) then returns the tile at exactly the planned size. Settings that are already 32-aligned keep their grid unchanged.
- The processing diagnostic is drawn over the enlarged baseline. The planner also exposes the same baseline without an overlay so workflows can display both views independently. Display downscaling must never alter tile metadata or generated images.

## Persistent Caches

Cache root:

```text
ComfyUI/user/<profile>/smart_upscaler_cache/
  prompts/
  esrgan/
```

Preprocessing keys include the source image, complete tile plan, selected enlargement method, optional tag, and—when used—a complete chunked fingerprint of the AI upscaler weights. The weight fingerprint is memoized per loaded model object, so correctness does not require re-hashing on every execution. `upscale_batch_size` and preview size are intentionally excluded because they do not change intended pixels.

Prompt keys (schema 2) include the actual capped image tensor seen by the VLM, complete instruction, tile reference/context, visible cache tag, and explicit Vision Model Cache ID. The connected caption-model input is lazy. On a valid hit, `check_lazy_status` does not request it, so the external VLM is not executed. Tensor storage is hashed through bounded memory views rather than making a second full-size byte copy.

`SmartCachedTilePromptGenerator` combines task-directed local generation, the per-tile cache, validation, and final resolver while keeping its `CLIP` caption-model input external and replaceable. General v10 uses deterministic `Managed by Prompt Director`; older generation values remain for backward loading only. Model thinking stays off so the output remains clean JSON. Empty/placeholder/black-background, generic-background, missing-target, template-echo, and user-confirmed false detections are never cached without repair. The retry retains the complete original task-aware instruction and adds only the validation failure; it never falls back to context-free captioning. False-detection and schema guards repair safe wording failures; conservative recovery prevents model formatting mistakes from aborting a full tile run. The node must preserve both lazy paths: valid cache hits skip generation, and `direct_user` bypasses generation regardless of cache mode.

Prompt-contract changes that affect spatial reasoning or operation intent require a cache-tag change. General v12 uses `qwen3vl_4b_fp8_local_target_v28_canonical_v12` for exact-tile target details and `qwen3vl_4b_fp8_master_scene_v21_canonical_v12` for the master scene prompt. Earlier tags (v19/v26 surface-only, v20/v27 v11) cannot be reused, so no caption written under an older contract can bypass the v12 validation. Whole-image `view` data is reduced to camera geometry. A tile receives only spatially matching surface, regional, and identity candidates—not the scene inventory—and must confirm them from its own pixels. The image model always receives a concise deterministic edit action before exact-tile target detail. Older source-only, locator-panel, standalone-scene, context-free retry, generic-region, or malformed captions cannot override the contract.

Cache modes:

- `read_write`: read an existing entry or generate and store a new one.
- `refresh`: evaluate the expensive source and replace the entry.
- `bypass`: evaluate normally without reading or writing disk cache.

Change the Vision Model Cache ID when replacing or updating the VLM, and change the prompt cache tag when changing the prompt contract, template behavior, or important hidden generation settings. Prompt entries are age/size bounded and touched on successful use; preprocessing entries are size/free-space bounded and touched on successful use. Cache write failure is non-fatal. Preprocessing cache entries preserve full tensor quality and can grow substantially. Deletion is safe; entries are reproducible.

## Memory and Speed

- Keep AI upscaler batch size at 1 as the dependable default. It is ignored by standard resizing.
- Lanczos is the sharpest conventional starting point when an AI upscaler rounds cars, lettering, or narrow geometry. Bicubic is gentler; bilinear and area are progressively softer.
- Preprocessing cache hits avoid AI inference or repeated resizing but still load the authoritative tile tensor batch from disk.
- Prompt cache hits skip unchanged VLM execution. `direct_user` skips both full-image and per-tile VLM branches without consulting the cache.
- Sampler controls are downstream of both caches. Changing denoise, steps, sampler, scheduler, CFG, seed, Flux model, or LoRA should reuse cached preprocessing and prompts.
- The node-level output-scale preview defaults to a 1600-pixel maximum dimension to avoid producing another full-resolution diagnostic tensor. The final workflow intentionally sets `preview_max_size=8192` so its processing-grid preview uses the actual output dimensions for normal images. Lower this value when very large previews create excessive system-memory use or temporary PNG overhead; it affects only the diagnostic display.
- Review with `single_tile` before changing to `all_tiles`. Partial merge fills unprocessed positions from the enlarged baseline.
- For faithful higher-denoise work, use `Match original appearance + structure`, Source Reference Strength 55-70, and Balanced detail freedom.
- For style transforms, use `Keep generated appearance; guide structure only` and lower Source Reference Strength. The original behavior aliases remain accepted internally for older workflows.
- `cross_tile_consistency` is style-safe relative alignment, not source color matching. It uses only shared generated overlap pixels, preserves the transformed batch's mean color, caps each tile offset, and runs only when all planned tiles are present. Start near 35; raise it only when broad surfaces visibly change hue or brightness across seams.

## Blend Diagnostic

- Magenta: vertical shared overlap.
- Cyan: horizontal shared overlap.
- Yellow gradients: the actual edge-mask feather ramps.
- White: the core-region join.
- Colored rectangles: expanded crops actually sent to preprocessing and diffusion.

Feather ramps are drawn from the outer edges of each shared overlap toward the interior, matching `_blend_mask`. Do not replace this with a decorative seam-centered gradient unless mask behavior also changes.

## Workflow Metadata

The final workflow must end in core ComfyUI `SaveImage`, which saves PNG and receives hidden `PROMPT` and `EXTRA_PNGINFO` inputs from Comfy execution. This embeds the graph and settings for drag-and-drop restoration. JPEG exports do not preserve this Comfy workflow payload.

The `All Prompts Viewer + Log` is an output-only diagnostic. Its readable report shows shared task/general-image text once and then one final prompt per tile. Its structured JSON retains local instructions, raw caption responses, resolver outputs, cache state, and tile references for deeper analysis. Reports are written beneath `ComfyUI/output/Smart-Upscaler/logs/`; `tile_prompt_audit_latest.*` is intentionally overwritten atomically while timestamped reports remain available for comparisons.

Do not replace the final saver with an arbitrary JPEG or image-library writer. If a custom saver is ever required, it must preserve Comfy's `prompt` and `extra_pnginfo` text chunks.

## Verification

Run:

```text
python -m unittest discover -s tests -v
```

Then verify against the live Comfy instance:

1. Every workflow node type exists in `/object_info`.
2. Every link references a valid source/output and target/input type.
3. The custom frontend extension returns HTTP 200.
4. A first run reports cache `WRITE`; an unchanged second run reports `HIT` without repeating preprocessing or the VLM.
5. A final PNG can be dragged into ComfyUI and restores the workflow.

Keep the workspace and live custom-node installation synchronized before browser testing.
