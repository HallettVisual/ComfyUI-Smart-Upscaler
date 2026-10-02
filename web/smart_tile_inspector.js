import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function imageUrl(image) {
  const query = new URLSearchParams({
    filename: image.filename,
    type: image.type,
    subfolder: image.subfolder || "",
  });
  return api.apiURL(`/view?${query.toString()}${app.getPreviewFormatParam()}${app.getRandParam()}`);
}

function makeButton(label, title) {
  const button = document.createElement("button");
  button.textContent = label;
  button.title = title;
  button.style.cssText = `
    width: 30px; height: 28px; border: 1px solid #555b66; border-radius: 4px;
    background: #30343b; color: #f1f3f5; cursor: pointer; font-size: 18px;
  `;
  return button;
}

// Tile prompt edits live on the Per-tile prompts node ("T006: prompt" lines),
// so they are saved with the workflow and visible in one place. The Inspector
// is only the comfortable way to write them.
const EDIT_LINE = /^\s*(T\d+)\s*:\s*(.*)$/i;

function parseEdits(text) {
  const edits = new Map();
  let current = null;
  for (const line of String(text || "").split(/\r?\n/)) {
    const match = line.match(EDIT_LINE);
    if (match) {
      current = match[1].toUpperCase();
      edits.set(current, match[2].trim());
    } else if (current && line.trim()) {
      edits.set(current, `${edits.get(current)} ${line.trim()}`.trim());
    }
  }
  return edits;
}

function formatEdits(edits) {
  return [...edits.entries()]
    .filter(([, prompt]) => prompt)
    .sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true }))
    .map(([tileId, prompt]) => `${tileId}: ${prompt}`)
    .join("\n");
}

function editWidgets() {
  return (app.graph?._nodes || [])
    .filter((node) => node.comfyClass === "SmartCachedTilePromptGenerator" || node.type === "SmartCachedTilePromptGenerator")
    .map((node) => node.widgets?.find((widget) => widget.name === "tile_prompt_edits"))
    .filter(Boolean);
}

function switchReuseOn() {
  for (const node of app.graph?._nodes || []) {
    if (node.comfyClass !== "SmartUnifiedPromptGuidance" && node.type !== "SmartUnifiedPromptGuidance") continue;
    const reuse = node.widgets?.find((widget) => widget.name === "prompt_reuse");
    const on = reuse?.options?.values?.find((value) => String(value).startsWith("On"));
    if (on) reuse.value = on;
  }
}

app.registerExtension({
  name: "ComfyUI.SmartUpscaler.TileInspector",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SmartTileInspector") return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalCreated?.apply(this, arguments);

      const container = document.createElement("div");
      container.style.cssText = `
        width: 100%; min-height: 640px; display: flex; flex-direction: column; gap: 8px;
        box-sizing: border-box; padding: 8px; overflow: hidden; background: #1b1d21;
        color: #edf0f2; font: 12px/1.4 system-ui, sans-serif;
      `;

      const toolbar = document.createElement("div");
      toolbar.style.cssText = "display:flex; align-items:center; gap:6px; width:100%;";
      const previous = makeButton("‹", "Previous tile");
      const next = makeButton("›", "Next tile");
      const selector = document.createElement("select");
      selector.title = "Select a tile by the same T-number shown on the processing grid";
      selector.style.cssText = `
        flex: 1; min-width: 0; height: 28px; border: 1px solid #555b66;
        border-radius: 4px; background: #262a30; color: #f5f6f7; padding: 0 7px;
      `;
      const counter = document.createElement("span");
      counter.style.cssText = "min-width:52px; text-align:right; color:#b9c0c8;";
      toolbar.append(previous, selector, next, counter);

      const comparison = document.createElement("div");
      comparison.style.cssText = `
        position: relative; width: 100%; min-height: 400px; flex: 1;
        overflow: hidden; border: 1px solid #4a5059; border-radius: 4px; background: #090a0c;
      `;
      const sourceImage = document.createElement("img");
      const generatedImage = document.createElement("img");
      for (const image of [sourceImage, generatedImage]) {
        image.draggable = false;
        image.style.cssText = `
          position:absolute; inset:0; width:100%; height:100%; object-fit:contain;
          user-select:none; pointer-events:none;
        `;
      }
      const divider = document.createElement("div");
      divider.style.cssText = `
        position:absolute; top:0; bottom:0; left:50%; width:2px; margin-left:-1px;
        background:#ffffff; box-shadow:0 0 0 1px #00000099; pointer-events:none;
      `;
      const sourceLabel = document.createElement("span");
      sourceLabel.textContent = "BEFORE";
      sourceLabel.style.cssText = "position:absolute;left:8px;top:8px;padding:3px 6px;background:#000a;color:#fff;border-radius:3px;";
      const generatedLabel = document.createElement("span");
      generatedLabel.textContent = "AFTER";
      generatedLabel.style.cssText = "position:absolute;right:8px;top:8px;padding:3px 6px;background:#000a;color:#fff;border-radius:3px;";
      comparison.append(sourceImage, generatedImage, divider, sourceLabel, generatedLabel);

      const split = document.createElement("input");
      split.type = "range";
      split.min = "0";
      split.max = "100";
      split.value = "50";
      split.title = "Move the before/after divider";
      split.style.cssText = "width:100%; margin:0; accent-color:#e8edf2;";

      const promptTitle = document.createElement("div");
      promptTitle.style.cssText = "font-weight:600;color:#ffd66b;";
      const prompt = document.createElement("textarea");
      prompt.spellcheck = false;
      prompt.title = "The exact prompt this tile was drawn with. Change it, then press 'Use my edit next run'.";
      prompt.style.cssText = `
        min-height: 92px; height: 120px; max-height: 240px; margin: 0; padding: 8px; resize: vertical;
        border: 1px solid #444a53; border-radius: 4px; background: #121418; color: #e8ebee;
        font: 12px/1.45 system-ui, sans-serif; box-sizing: border-box; width: 100%;
      `;
      prompt.placeholder = "Queue the workflow to inspect a tile and its exact prompt.";

      const editBar = document.createElement("div");
      editBar.style.cssText = "display:flex; align-items:center; gap:6px; width:100%;";
      const saveEdit = document.createElement("button");
      saveEdit.textContent = "Use my edit next run";
      saveEdit.title = "Stores this prompt for this tile on the Per-tile prompts node and turns on Reuse saved prompts. Next run, this tile skips the vision model and uses your text exactly.";
      const dropEdit = document.createElement("button");
      dropEdit.textContent = "Back to the model's prompt";
      dropEdit.title = "Removes your edit for this tile; the next run writes its prompt as usual.";
      for (const button of [saveEdit, dropEdit]) {
        button.style.cssText = `
          height: 28px; padding: 0 10px; border: 1px solid #555b66; border-radius: 4px;
          background: #30343b; color: #f1f3f5; cursor: pointer;
        `;
      }
      const editStatus = document.createElement("span");
      editStatus.style.cssText = "flex:1; min-width:0; color:#b9c0c8; text-align:right;";
      editBar.append(saveEdit, dropEdit, editStatus);
      container.append(toolbar, comparison, split, promptTitle, prompt, editBar);
      // Typing in the box must not trigger canvas shortcuts.
      prompt.addEventListener("keydown", (event) => event.stopPropagation());

      this.addDOMWidget("tile_inspector", "smart_tile_inspector", container, {
        serialize: false,
        getMinHeight: () => 700,
        getMaxHeight: () => Math.max(700, this.size?.[1] - 20 || 700),
      });
      this.setSize([760, 820]);

      this.smartTileInspector = { records: [], index: 0 };
      const render = () => {
        const records = this.smartTileInspector.records;
        if (!records.length) return;
        const index = Math.max(0, Math.min(this.smartTileInspector.index, records.length - 1));
        this.smartTileInspector.index = index;
        const record = records[index];
        selector.value = String(index);
        counter.textContent = `${index + 1} / ${records.length}`;
        sourceImage.src = imageUrl(record.source);
        generatedImage.src = imageUrl(record.generated);
        promptTitle.textContent = `${record.tile_id} | ${record.position} | row ${record.row}, col ${record.column}`;
        prompt.value = record.prompt;
        showEditStatus(record.tile_id);
        this.setDirtyCanvas(true, true);
      };
      const showEditStatus = (tileId) => {
        const widgets = editWidgets();
        if (!widgets.length) {
          editStatus.textContent = "Add the Per-tile prompts node to save edits.";
          return;
        }
        editStatus.textContent = parseEdits(widgets[0].value).has(tileId)
          ? "Your edit is saved for this tile."
          : "Prompt written by the model.";
      };
      const changeEdit = (prompt_text) => {
        const record = this.smartTileInspector.records[this.smartTileInspector.index];
        const widgets = editWidgets();
        if (!record || !widgets.length) {
          showEditStatus(record?.tile_id);
          return;
        }
        for (const widget of widgets) {
          const edits = parseEdits(widget.value);
          if (prompt_text) edits.set(record.tile_id, prompt_text);
          else edits.delete(record.tile_id);
          widget.value = formatEdits(edits);
        }
        if (prompt_text) switchReuseOn();
        editStatus.textContent = prompt_text
          ? "Saved. The next run uses your text for this tile (Reuse is now On)."
          : "Removed. The next run writes this tile's prompt as usual.";
        app.graph.setDirtyCanvas(true, true);
      };
      saveEdit.addEventListener("click", () => {
        changeEdit(prompt.value.replace(/\s+/g, " ").trim());
      });
      dropEdit.addEventListener("click", () => changeEdit(""));
      const updateSplit = () => {
        const value = Number(split.value);
        generatedImage.style.clipPath = `inset(0 0 0 ${value}%)`;
        divider.style.left = `${value}%`;
      };
      split.addEventListener("input", updateSplit);
      selector.addEventListener("change", () => {
        this.smartTileInspector.index = Number(selector.value);
        render();
      });
      previous.addEventListener("click", () => {
        const count = this.smartTileInspector.records.length;
        if (!count) return;
        this.smartTileInspector.index = (this.smartTileInspector.index - 1 + count) % count;
        render();
      });
      next.addEventListener("click", () => {
        const count = this.smartTileInspector.records.length;
        if (!count) return;
        this.smartTileInspector.index = (this.smartTileInspector.index + 1) % count;
        render();
      });
      updateSplit();
      this.renderSmartTileInspector = render;
      this.smartTileSelector = selector;
    };

    const originalExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      originalExecuted?.apply(this, arguments);
      let records = message?.smart_tile_inspector || [];
      if (records.length === 1 && Array.isArray(records[0])) records = records[0];
      this.smartTileInspector.records = records;
      this.smartTileInspector.index = 0;
      this.smartTileSelector.replaceChildren();
      for (const [index, record] of records.entries()) {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = `${record.tile_id} | ${record.position} | row ${record.row}, col ${record.column}`;
        this.smartTileSelector.appendChild(option);
      }
      this.renderSmartTileInspector();
    };
  },
});
