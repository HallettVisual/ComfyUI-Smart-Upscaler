import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const COMPLETE_PRESET_STORAGE_KEY = "smart-upscaler-complete-prompt-presets-v1";
const PRESET_ENDPOINT = "/smart_upscaler/prompt_presets";
const PRESET_PLACEHOLDER = "Choose a preset…";
const EDIT_HEADER = "── EDIT MODELS (Klein, Qwen Edit) ──";
const DENOISE_HEADER = "── DENOISE MODELS (Z-Turbo, SDXL, Flux) ──";
const SAVED_HEADER = "── YOUR SAVED PRESETS ──";
const EDIT_STYLE = "Instruction edit (Klein, Qwen Edit)";
const DENOISE_STYLE = "Plain description (SDXL, Flux, denoise)";
const DEFAULT_DETAIL = "Adaptive by Tile (recommended)";
const DEFAULT_SUFFIX = "Fine detail";
const DEFAULT_COLORS = "Off - change nothing (default)";

// Plain-language labels. The widget ORDER is fixed - saved workflows apply
// widget values by position, so nothing here may be reordered. Instead the
// names carry the grouping: 1-3 describe the job, 4-5 shape every tile prompt,
// 6 is a rare repair that is a true bypass unless it is switched on, and 7
// reuses saved prompts for an edited copy of a picture already run.
const WIDGET_LABELS = {
  instructions: "1. Instructions - what this job is",
  user_request: "2. Your request (optional)",
  known_false_detections: "3. Not in this image (optional)",
  prompt_suffix: "4. Tile prompts: extra words at the end",
  tile_detail: "5. Tile prompts: how much detail",
  tile_colors: "6. Repair: delete color names (rare)",
  prompt_reuse: "7. Reuse saved prompts (edited copy, same size)",
};
// These bodies mirror UNIFIED_TASK_INSTRUCTIONS in nodes/universal_prompting.py.
// Keep the two in sync so "Load Complete Preset" matches the node defaults.
const COMPLETE_PRESETS = {
  "Google Image Enhance": {
    family: "edit",
    instructions: `TASK: Google Image Enhance

Turn this Google Earth image into a realistic photograph. Keep everything exactly where it is - the same
buildings, streets, layout, and camera angle - and just make it look like a real, sharp photo.

Anything that looks broken, melted, smeared, or like debris is a capture glitch, not real damage: fix it
into how it should really look. Don't add anything that isn't clearly there, and never assume what kind
of place this is - describe only what is visible. Any big continuous area the image really has should
look the same everywhere it appears.`,
    userRequest: "",
    falseDetections: "",
  },
  "Style Transfer": {
    family: "edit",
    instructions: `TASK: Style Transfer

Restyle this image into the look you describe in User Request (for example a painting style, a film look,
or a material change). Keep the scene, shapes, and layout the same - only the style changes.

Apply the same look consistently across the whole image, and don't add objects or effects that aren't
actually there.`,
    userRequest: "",
    falseDetections: "",
  },
  "Time of Day": {
    family: "edit",
    instructions: `TASK: Time of Day

Change this image to the time of day you describe in User Request (for example day to night, or golden
hour). Keep everything in place - same buildings, layout, and camera angle - and only change the light,
color, and mood.

Don't invent new lights, reflections, or objects unless they are actually visible in the image.`,
    userRequest: "",
    falseDetections: "",
  },
  "Upscale / Detailer": {
    family: "edit",
    instructions: `TASK: Upscale / Detailer

Sharpen this image and add realistic detail without changing it. Keep the same content, colors, lighting,
and layout - just recover clean, believable detail.

Blur is part of the photo: out-of-focus areas stay just as soft, and only what is sharp gets sharper.
Thin real details that cross blurred areas are content, not blur, however fine: name them and keep
every one. If something is too blurry or small to identify for certain, leave
it as soft shapes and colors - never guess what it might be. Don't redesign anything or add objects
that aren't there.`,
    userRequest: "",
    falseDetections: "",
  },
};

function browserPresets(storageKey) {
  try {
    const value = JSON.parse(localStorage.getItem(storageKey) || "{}");
    return value && typeof value === "object" ? value : {};
  } catch {
    return {};
  }
}

async function fetchDiskPresets() {
  const response = await api.fetchApi(PRESET_ENDPOINT);
  if (!response.ok) {
    throw new Error(`Could not load Smart Upscaler presets (${response.status}).`);
  }
  const payload = await response.json();
  return {
    complete: payload.complete || {},
    legacy: payload.legacy || {},
    // The built-in pack ships as presets/task_presets.json on the server; the
    // embedded COMPLETE_PRESETS below are only the fallback when it is missing.
    builtin: payload.builtin || {},
  };
}

async function writeDiskPreset(kind, name, preset) {
  const response = await api.fetchApi(PRESET_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind, name, preset }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Could not save Smart Upscaler preset (${response.status}).`);
  }
  return payload.store || fetchDiskPresets();
}

async function migrateBrowserPresets(kind, storageKey, diskPresets) {
  const existing = diskPresets[kind] || {};
  const browserSaved = browserPresets(storageKey);
  let changed = false;
  for (const [name, preset] of Object.entries(browserSaved)) {
    if (!existing[name] && preset && typeof preset === "object") {
      await writeDiskPreset(kind, name, preset);
      changed = true;
    }
  }
  return changed ? fetchDiskPresets() : diskPresets;
}

function presetFamily(preset) {
  return String(preset?.family || "").toLowerCase() === "denoise" ? "denoise" : "edit";
}

// One list, split by model family with separator rows: edit-model presets,
// denoise-model presets, then the user's own saved presets.
function presetChoices(builtIns, saved) {
  const editNames = [];
  const denoiseNames = [];
  for (const [name, preset] of Object.entries(builtIns)) {
    (presetFamily(preset) === "denoise" ? denoiseNames : editNames).push(name);
  }
  const choices = [PRESET_PLACEHOLDER];
  if (editNames.length) {
    choices.push(EDIT_HEADER, ...editNames.map((name) => `Built-in: ${name}`));
  }
  if (denoiseNames.length) {
    choices.push(DENOISE_HEADER, ...denoiseNames.map((name) => `Built-in: ${name}`));
  }
  const savedNames = Object.keys(saved || {}).sort();
  if (savedNames.length) {
    choices.push(SAVED_HEADER, ...savedNames.map((name) => `Saved: ${name}`));
  }
  return choices;
}

function selectedPreset(value, builtIns, saved) {
  const choice = String(value || "");
  if (choice.startsWith("Built-in: ")) {
    return builtIns[choice.slice("Built-in: ".length)];
  }
  if (choice.startsWith("Saved: ")) {
    return saved?.[choice.slice("Saved: ".length)];
  }
  return null; // placeholder and separator rows select nothing
}

function refreshPresetCombo(widget, builtIns, saved) {
  widget.options.values = presetChoices(builtIns, saved);
  widget.value = PRESET_PLACEHOLDER;
}

function hideWidget(widget) {
  if (!widget || widget.type === "hidden") return;
  widget.origType = widget.type;
  widget.origComputeSize = widget.computeSize;
  widget.computeSize = () => [0, -4];
  widget.type = "hidden";
}

app.registerExtension({
  name: "ComfyUI.SmartUpscaler.CompletePromptPresets",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SmartUnifiedPromptGuidance") return;

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      originalConfigure?.apply(this, arguments);
      // Graphs saved before the reuse control existed hand it the Save
      // button's empty slot; show the real default instead of a blank value.
      const reuse = this.widgets?.find((widget) => widget.name === "prompt_reuse");
      const choices = reuse?.options?.values;
      if (reuse && Array.isArray(choices) && !choices.includes(reuse.value)) {
        reuse.value = choices[0];
      }
    };

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalCreated?.apply(this, arguments);
      const instructions = this.widgets?.find((widget) => widget.name === "instructions");
      const request = this.widgets?.find((widget) => widget.name === "user_request");
      const falseDetections = this.widgets?.find((widget) => widget.name === "known_false_detections");
      if (!instructions || !request || !falseDetections) return;
      // The sampler style is decided by the preset's model family, not a
      // separate on-screen dial; the input stays real but invisible.
      hideWidget(this.widgets?.find((widget) => widget.name === "sampler_prompt_style"));
      for (const widget of this.widgets || []) {
        if (WIDGET_LABELS[widget.name]) widget.label = WIDGET_LABELS[widget.name];
      }

      let savedPresets = {};
      let builtInPresets = COMPLETE_PRESETS;
      const loadPreset = this.addWidget(
        "combo",
        "Load Preset",
        PRESET_PLACEHOLDER,
        (value) => {
          const selected = selectedPreset(value, builtInPresets, savedPresets);
          if (!selected) return;
          // A preset fills EVERY field. A value the preset does not carry goes
          // back to its default rather than lingering from the last preset -
          // a stuck detail level or suffix after switching model family is the
          // hardest kind of wrong result to spot.
          instructions.value = selected.instructions || "";
          request.value = selected.userRequest || "";
          falseDetections.value = selected.falseDetections || "";
          const suffixWidget = this.widgets?.find((widget) => widget.name === "prompt_suffix");
          if (suffixWidget) {
            suffixWidget.value = String(selected.promptSuffix ?? DEFAULT_SUFFIX);
          }
          const detailWidget = this.widgets?.find((widget) => widget.name === "tile_detail");
          if (detailWidget) {
            detailWidget.value = selected.tileDetail || DEFAULT_DETAIL;
          }
          const styleWidget = this.widgets?.find((widget) => widget.name === "sampler_prompt_style");
          if (styleWidget) {
            styleWidget.value =
              selected.samplerStyle ||
              (presetFamily(selected) === "denoise" ? DENOISE_STYLE : EDIT_STYLE);
          }
          const colorWidget = this.widgets?.find((widget) => widget.name === "tile_colors");
          if (colorWidget) {
            colorWidget.value = selected.tileColors || DEFAULT_COLORS;
          }
          this.setDirtyCanvas(true, true);
        },
        { values: presetChoices(builtInPresets, savedPresets) },
      );
      loadPreset.label = "Load Preset (fills in everything above)";
      loadPreset.serialize = false;
      loadPreset.options.serialize = false;

      const refreshFromDisk = async () => {
        let store = await fetchDiskPresets();
        store = await migrateBrowserPresets(
          "complete",
          COMPLETE_PRESET_STORAGE_KEY,
          store,
        );
        savedPresets = store.complete || {};
        if (store.builtin && Object.keys(store.builtin).length) {
          builtInPresets = store.builtin;
        }
        refreshPresetCombo(loadPreset, builtInPresets, savedPresets);
        this.setDirtyCanvas(true, true);
      };
      refreshFromDisk().catch((error) => console.error("Smart Upscaler presets:", error));

      this.addWidget("button", "Save My Preset to Disk", null, async () => {
        const name = window.prompt("Name this complete Smart Upscaler preset:");
        if (!name?.trim()) return;
        const styleWidget = this.widgets?.find((widget) => widget.name === "sampler_prompt_style");
        const detailWidget = this.widgets?.find((widget) => widget.name === "tile_detail");
        const suffixWidget = this.widgets?.find((widget) => widget.name === "prompt_suffix");
        const colorWidget = this.widgets?.find((widget) => widget.name === "tile_colors");
        const savedPreset = {
          instructions: String(instructions.value || ""),
          userRequest: String(request.value || ""),
          falseDetections: String(falseDetections.value || ""),
          promptSuffix: String(suffixWidget?.value ?? ""),
          tileDetail: String(detailWidget?.value || DEFAULT_DETAIL),
          samplerStyle: String(styleWidget?.value || EDIT_STYLE),
          tileColors: String(colorWidget?.value || DEFAULT_COLORS),
          family: String(styleWidget?.value || "") === DENOISE_STYLE ? "denoise" : "edit",
        };
        try {
          const store = await writeDiskPreset("complete", name.trim(), savedPreset);
          savedPresets = store.complete || {};
          refreshPresetCombo(loadPreset, builtInPresets, savedPresets);
          loadPreset.value = `Saved: ${name.trim()}`;
          this.setDirtyCanvas(true, true);
        } catch (error) {
          window.alert(error.message || String(error));
        }
      });

      // Compact default: four text boxes share a modest node; saved workflows
      // restore their own size after creation, so this only shapes new nodes.
      const minimum = this.computeSize();
      this.setSize([Math.max(minimum[0], 420), minimum[1]]);
    };
  },
});
