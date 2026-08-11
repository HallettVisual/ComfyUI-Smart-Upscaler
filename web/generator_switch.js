import { app } from "../../scripts/app.js";

// Generator Switch polish:
// - generator inputs grow as chains are connected (always exactly one spare)
// - the dropdown lists connected chains by NAME ("3: Flux 1"), read from the
//   slot label or the connected node's title; python parses the leading number
const MAX_GENERATORS = 10;
const GENERATOR_PREFIX = "generator_";
const LEGACY_PREFIX = "engine_";
const NUMBER_INPUT = "generator_number";

function migrateLegacyNames(node) {
  // Graphs saved before the "engine" -> "generator" rename carry engine_N
  // sockets, some with live links. Rename in place: the input objects and the
  // numeric target_slot indexes that LiteGraph links point at stay untouched,
  // so no wire is lost.
  for (const input of node.inputs || []) {
    const name = String(input.name || "");
    // "engine" is the dropdown itself, which newer frontends also list here.
    const renamed = name.startsWith(LEGACY_PREFIX)
      ? `${GENERATOR_PREFIX}${name.slice(LEGACY_PREFIX.length)}`
      : name === "engine"
        ? "generator"
        : null;
    if (renamed === null) continue;
    if (input.label === name) input.label = renamed;
    input.name = renamed;
  }
}

function generatorWidget(node) {
  // Old graphs may still present the widget under its previous name.
  return node.widgets?.find(
    (candidate) => candidate.name === "generator" || candidate.name === "engine",
  );
}

function generatorSlots(node) {
  return (node.inputs || [])
    .map((input, index) => ({ input, index }))
    .filter(
      ({ input }) =>
        input.name?.startsWith(GENERATOR_PREFIX) && input.name !== NUMBER_INPUT,
    );
}

function slotNumber(input) {
  return parseInt(input.name.slice(GENERATOR_PREFIX.length), 10);
}

function containingGroupTitle(graph, origin) {
  // The largest titled group holding the upstream node - generator chains live
  // in top-level groups named after the model ("Flux 1", "SDXL"), so that title
  // beats the feeding node's own name (usually just "VAE Decode").
  const groups = graph?._groups || graph?.groups || [];
  const centerX = origin.pos[0] + (origin.size?.[0] || 0) / 2;
  const centerY = origin.pos[1] + (origin.size?.[1] || 0) / 2;
  let best = null;
  let bestArea = -1;
  for (const group of groups) {
    const bounds = group._bounding || group.bounding;
    if (!bounds || !String(group.title || "").trim()) continue;
    const [x, y, width, height] = bounds;
    if (centerX < x || centerY < y || centerX > x + width || centerY > y + height) continue;
    const area = width * height;
    if (area > bestArea) {
      bestArea = area;
      best = group;
    }
  }
  return best ? String(best.title).trim() : null;
}

function resolveReroute(graph, node) {
  // A KJNodes GET reroute hides the real chain: the node sits next to the
  // switch, not inside the generator's group, so "Get_Z-Turbo" is all we would
  // see. Follow it back to the matching SetNode, which does live in the group.
  if (String(node?.type || "") !== "GetNode") return node;
  const key = String(node.widgets?.[0]?.value ?? "").trim();
  if (!key) return node;
  const nodes = graph?._nodes || graph?.nodes || [];
  const setter = nodes.find(
    (candidate) =>
      String(candidate?.type || "") === "SetNode" &&
      String(candidate.widgets?.[0]?.value ?? "").trim() === key,
  );
  return setter || node;
}

function cleanTitle(value) {
  return String(value || "")
    .replace(/^(?:Get|Set)[_\s]+/i, "")
    .replace(/_/g, " ")
    .trim();
}

function chainLabel(node, input) {
  if (input.label && input.label !== input.name) return input.label;
  if (input.link != null) {
    const link = node.graph?.links?.[input.link];
    const linked = link ? node.graph.getNodeById(link.origin_id) : null;
    const origin = linked ? resolveReroute(node.graph, linked) : null;
    if (origin) {
      return (
        containingGroupTitle(node.graph, origin) ||
        cleanTitle(origin.title) ||
        cleanTitle(linked.title) ||
        origin.type
      );
    }
  }
  return null;
}

function refreshGeneratorChoices(node) {
  const widget = generatorWidget(node);
  if (!widget) return;
  const connected = generatorSlots(node).filter(({ input }) => input.link != null);
  const source = connected.length ? connected : generatorSlots(node);
  const values = source.map(({ input }) => {
    const number = slotNumber(input);
    const label = chainLabel(node, input);
    return label ? `${number}: ${label}` : `Generator ${number}`;
  });
  widget.options.values = values;
  // Keep the same SLOT selected across relabeling; fall back to the first.
  const current = parseInt(String(widget.value).match(/\d+/)?.[0] ?? "1", 10);
  widget.value =
    values.find((value) => parseInt(value.match(/\d+/)?.[0] ?? "0", 10) === current) ||
    values[0] ||
    "Generator 1";
}

function normalizeGeneratorInputs(node) {
  // Keep every connected slot, numbered order intact, plus exactly one spare.
  const slots = generatorSlots(node);
  let highestConnected = 0;
  for (const { input } of slots) {
    if (input.link != null) highestConnected = Math.max(highestConnected, slotNumber(input));
  }
  const wanted = Math.min(Math.max(highestConnected + 1, 2), MAX_GENERATORS);
  for (const { input, index } of [...slots].reverse()) {
    if (slotNumber(input) > wanted && input.link == null) node.removeInput(index);
  }
  const present = new Set(generatorSlots(node).map(({ input }) => slotNumber(input)));
  for (let number = 1; number <= wanted; number += 1) {
    if (!present.has(number)) node.addInput(`${GENERATOR_PREFIX}${number}`, "*");
  }
  // generator_number stays the last input; generator slots stay in numeric order.
  node.inputs.sort((a, b) => {
    const rank = (input) =>
      input.name === NUMBER_INPUT
        ? MAX_GENERATORS + 1
        : input.name?.startsWith(GENERATOR_PREFIX)
          ? slotNumber(input)
          : -1;
    return rank(a) - rank(b);
  });
  // LiteGraph links store the numeric target slot. Reordering only the input
  // array leaves those indexes stale and can silently route a connected model
  // chain into the wrong generator after reload.
  node.inputs.forEach((input, index) => {
    if (input.link == null) return;
    const link = node.graph?.links?.[input.link];
    if (link) link.target_slot = index;
  });
  node.setSize(node.computeSize());
}

app.registerExtension({
  name: "ComfyUI.SmartUpscaler.GeneratorSwitch",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SmartModelEngineSwitch") return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalCreated?.apply(this, arguments);
      migrateLegacyNames(this);
      normalizeGeneratorInputs(this);
      refreshGeneratorChoices(this);
    };

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      originalConfigure?.apply(this, arguments);
      migrateLegacyNames(this);
      normalizeGeneratorInputs(this);
      refreshGeneratorChoices(this);
    };

    const originalConnections = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo, ioSlot) {
      originalConnections?.apply(this, arguments);
      if (ioSlot?.name?.startsWith(GENERATOR_PREFIX) && ioSlot.name !== NUMBER_INPUT) {
        normalizeGeneratorInputs(this);
        refreshGeneratorChoices(this);
      }
    };
  },
});
