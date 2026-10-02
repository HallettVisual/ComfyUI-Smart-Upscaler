import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Live readout: the per-tile prompt node sends each tile's finished prompt as
// soon as it is written (TILE_PROMPT_EVENT in nodes/cache.py). The complete
// log replaces this text when the Prompt log node itself runs at the end.
const TILE_PROMPT_EVENT = "smart_upscaler.tile_prompt";
let liveTiles = 0;

function auditViews() {
  return (app.graph?._nodes || [])
    .map((node) => node.smartPromptAudit)
    .filter(Boolean);
}

api.addEventListener("execution_start", () => {
  liveTiles = 0;
});

api.addEventListener(TILE_PROMPT_EVENT, ({ detail }) => {
  liveTiles += 1;
  const entry = `${detail.tile_id} | ${detail.position}\n${detail.prompt}\n[${detail.status}]\n\n`;
  for (const view of auditViews()) {
    if (liveTiles === 1) {
      view.report.textContent =
        "LIVE - each tile's prompt appears here as it is written.\n" +
        "The complete log replaces this when the run finishes.\n\n";
    }
    view.report.textContent += entry;
    view.report.scrollTop = view.report.scrollHeight;
    view.status.textContent = `Writing tile prompts... ${liveTiles} done (latest ${detail.tile_id})`;
  }
  app.graph?.setDirtyCanvas(true, true);
});

app.registerExtension({
  name: "ComfyUI.SmartUpscaler.PromptAuditViewer",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SmartTilePromptAuditLog") return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalCreated?.apply(this, arguments);

      const container = document.createElement("div");
      container.style.cssText = `
        width:100%; height:100%; min-height:560px; display:flex; flex-direction:column;
        gap:7px; box-sizing:border-box; padding:8px; background:#17191d; color:#edf0f2;
        font:12px/1.45 system-ui,sans-serif;
      `;
      const toolbar = document.createElement("div");
      toolbar.style.cssText = "display:flex;gap:6px;align-items:center;";
      const copy = document.createElement("button");
      copy.textContent = "Copy all text";
      copy.style.cssText = "padding:5px 9px;background:#343a43;color:#fff;border:1px solid #59616d;border-radius:4px;cursor:pointer;";
      const status = document.createElement("span");
      status.textContent = "Queue the workflow to display the complete prompt process.";
      status.style.cssText = "flex:1;color:#bfc6cf;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
      toolbar.append(copy, status);

      const report = document.createElement("pre");
      report.textContent = "No audit has run yet.";
      report.style.cssText = `
        flex:1; min-height:500px; margin:0; padding:10px; overflow:auto; resize:none;
        white-space:pre-wrap; overflow-wrap:anywhere; border:1px solid #484e58;
        border-radius:4px; background:#0f1114; color:#e8ebef; font:12px/1.45 Consolas,monospace;
      `;
      copy.onclick = async () => {
        await navigator.clipboard.writeText(report.textContent || "");
        const old = copy.textContent;
        copy.textContent = "Copied";
        setTimeout(() => (copy.textContent = old), 1200);
      };
      container.append(toolbar, report);
      this.addDOMWidget("prompt_audit_viewer", "prompt_audit_viewer", container, {
        serialize: false,
        getMinHeight: () => 580,
      });
      this.setSize([900, 720]);
      this.smartPromptAudit = { report, status };
    };

    const originalExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      originalExecuted?.apply(this, arguments);
      if (!this.smartPromptAudit) return;
      const text = Array.isArray(message?.prompt_audit_text)
        ? message.prompt_audit_text[0]
        : message?.prompt_audit_text;
      const path = Array.isArray(message?.readable_log_path)
        ? message.readable_log_path[0]
        : message?.readable_log_path;
      if (text) this.smartPromptAudit.report.textContent = text;
      this.smartPromptAudit.status.textContent = path || "Prompt audit displayed.";
      this.setDirtyCanvas(true, true);
    };
  },
});
