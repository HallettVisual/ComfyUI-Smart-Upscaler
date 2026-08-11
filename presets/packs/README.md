# Drop-in preset packs

Any `*.json` file in this folder is merged into the **Load Preset** dropdown when
ComfyUI starts, on top of the presets that ship with the node pack.

Use it for add-on packs you have bought or written yourself. Nothing here is
overwritten when you update Smart Upscaler, and nothing here is published.

A pack file looks exactly like `presets/task_presets.json`:

```json
{
  "version": 1,
  "presets": {
    "My Preset Name": {
      "instructions": "TASK: Upscale / Detailer\n\n...",
      "userRequest": "",
      "falseDetections": "",
      "tileDetail": "Adaptive by Tile (recommended)",
      "promptSuffix": "",
      "tileColors": "Off - change nothing (default)",
      "samplerStyle": "Instruction edit (Klein, Qwen Edit)",
      "family": "edit"
    }
  }
}
```

Packs load in filename order after the shipped set, so a pack can deliberately
replace a shipped preset by reusing its name. Restart ComfyUI after adding one.
