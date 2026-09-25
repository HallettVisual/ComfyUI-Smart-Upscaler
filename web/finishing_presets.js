import { app } from "../../scripts/app.js";

// Quick Presets on the finishing nodes write their values INTO the visible
// dials, then stay selected as a label. The user sees exactly what
// the preset chose and fine-tunes from there. These mappings mirror
// FINISH_PRESETS in nodes/finalize.py and COLOR_MATCH_PRESETS in
// nodes/fidelity.py - keep them in sync.

const MANUAL = "Manual - I set the dials myself";

const FINISH_PRESETS = {
  "Photo upscale, seams hidden (start here)": {
    reference_mode: "Stay close to the original photo",
    structure_preservation: 50,
    detail_support: "Keep more new detail",
    cross_tile_consistency: 60,
    consistency_mode: "Fade the fix toward the edge that disagrees (best for seams)"
  },
  "Photo upscale, hug the original": {
    reference_mode: "Stay close to the original photo",
    structure_preservation: 55,
    detail_support: "Balanced",
    cross_tile_consistency: 35,
    consistency_mode: "Shift the whole tile evenly"
  },
  "I changed the look (style, time of day)": {
    reference_mode: "Keep the new look (only fix warped shapes)",
    structure_preservation: 20,
    detail_support: "Balanced",
    cross_tile_consistency: 35,
    consistency_mode: "Shift the whole tile evenly"
  },
  "I changed the look, seams hidden": {
    reference_mode: "Keep the new look (only fix warped shapes)",
    structure_preservation: 20,
    detail_support: "Balanced",
    cross_tile_consistency: 60,
    consistency_mode: "Fade the fix toward the edge that disagrees (best for seams)"
  },
  "Let the model add the most detail": {
    reference_mode: "Keep the new look (only fix warped shapes)",
    structure_preservation: 10,
    detail_support: "Keep more new detail",
    cross_tile_consistency: 35,
    consistency_mode: "Shift the whole tile evenly"
  },
};

const COLOR_MATCH_PRESETS = {
  "Automatic - original colors unless the task changes the look (recommended)": {
    color_match_method: "automatic",
    color_match_strength: 100,
  },
  "Original colors, no tile seams": {
    color_match_method: "original_colors",
    color_match_strength: 100,
  },
  "No color change (style/lighting edits)": {
    color_match_method: "none",
    color_match_strength: 0,
  },
};

function attachPresetApplier(nodeType, presetWidgetName, presets) {
  const originalCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    originalCreated?.apply(this, arguments);
    const presetWidget = this.widgets?.find(
      (widget) => widget.name === presetWidgetName,
    );
    if (!presetWidget) return;
    const originalCallback = presetWidget.callback;
    presetWidget.callback = (value, ...args) => {
      originalCallback?.call(presetWidget, value, ...args);
      const mapping = presets[String(value)];
      if (!mapping) return;
      for (const [name, target] of Object.entries(mapping)) {
        const widget = this.widgets?.find((item) => item.name === name);
        if (widget) widget.value = target;
      }
      // The preset name stays selected as a visible label of what was loaded;
      // the dials below hold the actual values and remain freely tunable.
      this.setDirtyCanvas(true, true);
    };
  };
}

app.registerExtension({
  name: "ComfyUI.SmartUpscaler.FinishingPresets",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "SmartTileFinalizer") {
      attachPresetApplier(nodeType, "finish_preset", FINISH_PRESETS);
    } else if (nodeData.name === "SmartTileColorMatch") {
      attachPresetApplier(nodeType, "color_preset", COLOR_MATCH_PRESETS);
    }
  },
});
