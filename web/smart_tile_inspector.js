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
      const prompt = document.createElement("pre");
      prompt.style.cssText = `
        min-height: 92px; max-height: 190px; margin: 0; padding: 8px; overflow: auto;
        white-space: pre-wrap; overflow-wrap: anywhere; border: 1px solid #444a53;
        border-radius: 4px; background: #121418; color: #e8ebee; font: 12px/1.45 system-ui, sans-serif;
      `;
      prompt.textContent = "Queue the workflow to inspect a tile and its exact prompt.";
      container.append(toolbar, comparison, split, promptTitle, prompt);

      this.addDOMWidget("tile_inspector", "smart_tile_inspector", container, {
        serialize: false,
        getMinHeight: () => 640,
        getMaxHeight: () => Math.max(640, this.size?.[1] - 20 || 640),
      });
      this.setSize([760, 760]);

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
        prompt.textContent = record.prompt;
        this.setDirtyCanvas(true, true);
      };
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
