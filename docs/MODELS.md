# Model downloads

Everything here is free. Put each file in the folder shown, then restart
ComfyUI. About **13 GB** in total.

This same list is also printed inside the workflow itself, in the note titled
**MODEL DOWNLOADS — read this first**.

---

## 1. The model that reads your picture

This is the part that makes Smart Upscaler different — it looks at your whole
image and writes the prompts.

`ComfyUI/models/text_encoders/`

| File | Size |
|---|---|
| [qwen3vl_4b_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) | 4.9 GB |

> **In the CLIPLoader, `type` must be set to `krea2`.**
> That setting is what loads the full Qwen3-VL wrapper, including the vision
> tower. Any other type loads a text-only model, the captioner runs blind, and
> every tile prompt comes back empty or generic. The workflow ships with a note
> next to that loader saying the same thing.

An 8B version also works if you have the VRAM — same repo family, same `krea2`
type. The 4B FP8 file above is the recommended balance.

---

## 2. The model that draws — Z-Image Turbo

### Diffusion model

`ComfyUI/models/diffusion_models/` — pick **one**:

| File | Size | Who it is for |
|---|---|---|
| [z_image_turbo_bf16.safetensors](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors) | 11.5 GB | works on any card |
| [z_image_turbo_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors) | 5.8 GB | smaller and faster |
| `z_image_turbo_nvfp4.safetensors` | 4.2 GB | **RTX 50-series only** — in the [same folder](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/diffusion_models) |

The workflow ships pointing at the **nvfp4** file. If you are not on a 50-series
card, open the **Load Diffusion Model** node and select the file you downloaded.

### Text encoder

`ComfyUI/models/text_encoders/`

| File | Size |
|---|---|
| [qwen_3_4b.safetensors](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) | 7.5 GB |

This is a **different model** from the vision one in section 1, and it must stay
on its own loader. Never share one CLIPLoader between captioning and sampler
conditioning.

### VAE

`ComfyUI/models/vae/`

| File | Size |
|---|---|
| [ae.safetensors](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors) | 0.3 GB |

---

## 3. Tile ControlNet — holds your layout steady

`ComfyUI/models/model_patches/`

The workflow is set to the **Tile** model from Alibaba's Z-Image-Turbo-Fun
ControlNet family. Find it on the
[alibaba-pai Hugging Face page](https://huggingface.co/alibaba-pai) — look for
`Z-Image-Turbo-Fun-Controlnet-Tile`.

The **Union** model works too. If that is what you have, just select it in the
`ModelPatchLoader` node:

- [Z-Image-Turbo-Fun-Controlnet-Union.safetensors](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors)

---

## 4. Optional — an ESRGAN-style enlarger

`ComfyUI/models/upscale_models/`

Only needed if you change the Tile Planner's first dropdown to
**AI upscaler model**. Any 4x model from [OpenModelDB](https://openmodeldb.info/)
works.

Out of the box the workflow uses **Lanczos**, which needs no model at all and
keeps the source geometry honest. Worth trying both on your own image — results
are cached, so the second attempt is fast.

---

## Where everything goes

```
ComfyUI/
└── models/
    ├── text_encoders/
    │   ├── qwen3vl_4b_fp8_scaled.safetensors    <- reads your picture
    │   └── qwen_3_4b.safetensors                <- Z-Image text encoder
    ├── diffusion_models/
    │   └── z_image_turbo_*.safetensors          <- pick one variant
    ├── vae/
    │   └── ae.safetensors
    ├── model_patches/
    │   └── Z-Image-Turbo-Fun-Controlnet-Tile-*.safetensors
    └── upscale_models/
        └── (optional 4x model)
```

---

## Custom nodes

Install all three from the ComfyUI Manager:

| Node pack | What it is for here |
|---|---|
| **ComfyUI-Smart-Upscaler** | this pack |
| **ComfyUI-KJNodes** | the `Set_` / `Get_` wires that keep the graph readable |
| **rgthree-comfy** | the before/after comparison sliders |

---

## VRAM

The workflow was built and tested on 16 GB. It fits because each stage loads,
finishes all its work, and unloads before the next one starts:

```
vision model (4.9 GB) → reads the picture, writes every tile prompt → unloaded
text encoder (7.5 GB) → encodes every prompt                        → unloaded
Z-Image + ControlNet (≈6 GB) → renders every tile
```

One model swap per stage, not per tile. Sampler tiles are 1024–1536 px, so the
working memory is small — it is the model weights that dominate.

If you are tight on memory:

- use the **int8** diffusion model instead of bf16
- lower **Prompt Detail Level** on the Prompt Director
- lower **max_tile_size** in the Tile Planner
