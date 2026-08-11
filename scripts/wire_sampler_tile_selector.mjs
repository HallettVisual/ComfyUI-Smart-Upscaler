import fs from "node:fs";
import path from "node:path";

const [sourcePath, destinationPath, tileValue = "1"] = process.argv.slice(2);
if (!sourcePath || !destinationPath) {
  throw new Error(
    "Usage: node wire_sampler_tile_selector.mjs <source.json> <destination.json> [tile-number]",
  );
}
if (path.resolve(sourcePath) === path.resolve(destinationPath)) {
  throw new Error("Source and destination must differ; this tool preserves the original workflow.");
}
const tileNumber = Math.max(1, Number.parseInt(tileValue, 10) || 1);
const workflow = JSON.parse(fs.readFileSync(sourcePath, "utf8"));
if (!Array.isArray(workflow.nodes) || !Array.isArray(workflow.links)) {
  throw new Error("The source is not a valid ComfyUI workflow.");
}
function oneNode(predicate, label, required = true) {
  const matches = workflow.nodes.filter(predicate);
  if (matches.length === 1) return matches[0];
  if (!required && matches.length === 0) return null;
  throw new Error(`Expected exactly one ${label}; found ${matches.length}.`);
}

function storedName(node) {
  const value = node.widgets_values;
  return String(Array.isArray(value) ? value[0] : value ?? "");
}

function inputSlot(node, name) {
  const slot = (node.inputs ?? []).findIndex((input) => input.name === name);
  if (slot < 0) throw new Error(`${node.type} ${node.id} is missing input ${name}.`);
  return slot;
}

function removeInputLink(node, name) {
  const slot = inputSlot(node, name);
  const linkId = node.inputs[slot].link;
  if (linkId == null) return slot;
  const link = workflow.links.find((candidate) => Number(candidate[0]) === Number(linkId));
  if (!link) throw new Error(`Input ${node.id}:${name} points to missing link ${linkId}.`);
  const origin = workflow.nodes.find((candidate) => Number(candidate.id) === Number(link[1]));
  const output = origin?.outputs?.[Number(link[2])];
  if (output && Array.isArray(output.links)) {
    output.links = output.links.filter((id) => Number(id) !== Number(linkId));
    if (output.links.length === 0) output.links = null;
  }
  workflow.links = workflow.links.filter(
    (candidate) => Number(candidate[0]) !== Number(linkId),
  );
  node.inputs[slot].link = null;
  return slot;
}

let nextLink = Math.max(
  Number(workflow.last_link_id ?? 0),
  ...workflow.links.map((link) => Number(link[0]) || 0),
);
function addLink(origin, originSlot, target, targetSlot, type) {
  const id = ++nextLink;
  workflow.links.push([id, origin.id, originSlot, target.id, targetSlot, type]);
  const output = origin.outputs[originSlot];
  if (!Array.isArray(output.links)) output.links = [];
  output.links.push(id);
  target.inputs[targetSlot].link = id;
  return id;
}

const director = oneNode(
  (node) => node.type === "SmartTileJobDirector",
  "SmartTileJobDirector",
);
const prompt = oneNode(
  (node) => node.type === "SmartCachedTilePromptGenerator",
  "SmartCachedTilePromptGenerator",
);
const finalizer = oneNode(
  (node) => node.type === "SmartTileFinalizer",
  "SmartTileFinalizer",
);
const colorMatch = oneNode(
  (node) => node.type === "SmartTileColorMatch",
  "SmartTileColorMatch",
  false,
);
const inspector = oneNode(
  (node) => node.type === "SmartTileInspector",
  "SmartTileInspector",
  false,
);
const setImage = oneNode(
  (node) => node.type === "SetNode" && storedName(node) === "Tiled_Image",
  "Set_Tiled_Image",
);
const setPositive = oneNode(
  (node) => node.type === "SetNode" && storedName(node) === "Positive_Prompts",
  "Set_Positive_Prompts",
);
const setNegative = oneNode(
  (node) => node.type === "SetNode" && storedName(node) === "Negative_Prompts",
  "Set_Negative_Prompts",
);
const setSeed = oneNode(
  (node) => node.type === "SetNode" && storedName(node) === "Seed",
  "Set_Seed",
);

const maximumNodeId = Math.max(
  Number(workflow.last_node_id ?? 0),
  ...workflow.nodes.map((node) => Number(node.id) || 0),
);
const selectors = workflow.nodes.filter(
  (node) => node.type === "SmartSamplerTileSelector",
);
if (selectors.length > 1) {
  throw new Error(`Expected at most one SmartSamplerTileSelector; found ${selectors.length}.`);
}
let selector = selectors[0];
if (selector) {
  // Preserve any selector outputs the user already connected inside a sampler
  // engine. Once the shared SET fan-out below is selected too, those direct
  // connections are safe and no longer broadcast against full lists.
  for (const name of [
    "tile_images",
    "positive_prompts",
    "negative_prompts",
    "tile_references",
    "tile_seeds",
  ]) {
    removeInputLink(selector, name);
  }
  selector.pos = [850, -1320];
  selector.size = [520, 350];
  selector.title = "Sampler Test Gate: Keep All Prompts, Run One T-Number";
  selector.widgets_values = ["One tile test", tileNumber];
} else {
  selector = {
    id: maximumNodeId + 1,
    type: "SmartSamplerTileSelector",
    pos: [850, -1320],
    size: [520, 350],
    flags: {},
    order: Math.max(...workflow.nodes.map((node) => Number(node.order) || 0)) + 1,
    mode: 0,
    inputs: [
      { name: "tile_images", type: "IMAGE", link: null },
      { name: "positive_prompts", type: "STRING", link: null },
      { name: "negative_prompts", type: "STRING", link: null },
      { name: "tile_references", type: "STRING", link: null },
      { name: "tile_seeds", type: "INT", link: null },
      { name: "processing_mode", type: "COMBO", link: null },
      { name: "tile_number", type: "INT", link: null },
    ],
    outputs: [
      { name: "tile_images", type: "IMAGE", links: null },
      { name: "positive_prompts", type: "STRING", links: null },
      { name: "negative_prompts", type: "STRING", links: null },
      { name: "tile_references", type: "STRING", links: null },
      { name: "tile_seeds", type: "INT", links: null },
    ],
    properties: { "Node name for S&R": "SmartSamplerTileSelector" },
    widgets_values: ["One tile test", tileNumber],
    title: "Sampler Test Gate: Keep All Prompts, Run One T-Number",
  };
  workflow.nodes.push(selector);
}

removeInputLink(setImage, setImage.inputs[0].name);
removeInputLink(setPositive, setPositive.inputs[0].name);
removeInputLink(setNegative, setNegative.inputs[0].name);
removeInputLink(setSeed, setSeed.inputs[0].name);
removeInputLink(finalizer, "tile_references");
if (colorMatch) removeInputLink(colorMatch, "source_tile");
if (inspector) {
  removeInputLink(inspector, "source_images");
  removeInputLink(inspector, "prompts");
  removeInputLink(inspector, "tile_references");
}

addLink(director, 0, selector, 0, "IMAGE");
addLink(prompt, 0, selector, 1, "STRING");
addLink(prompt, 1, selector, 2, "STRING");
addLink(prompt, 3, selector, 3, "STRING");
addLink(director, 3, selector, 4, "INT");
addLink(selector, 0, setImage, 0, "IMAGE");
addLink(selector, 1, setPositive, 0, "STRING");
addLink(selector, 2, setNegative, 0, "STRING");
addLink(selector, 3, finalizer, inputSlot(finalizer, "tile_references"), "STRING");
addLink(selector, 4, setSeed, 0, "INT");
if (colorMatch) {
  addLink(selector, 0, colorMatch, inputSlot(colorMatch, "source_tile"), "IMAGE");
}
if (inspector) {
  addLink(selector, 0, inspector, inputSlot(inspector, "source_images"), "IMAGE");
  addLink(selector, 1, inspector, inputSlot(inspector, "prompts"), "STRING");
  addLink(selector, 3, inspector, inputSlot(inspector, "tile_references"), "STRING");
}

if (Array.isArray(director.widgets_values) && director.widgets_values.length) {
  director.widgets_values[0] = "all_tiles";
}
workflow.last_node_id = Math.max(
  Number(workflow.last_node_id ?? 0),
  ...workflow.nodes.map((node) => Number(node.id) || 0),
);
workflow.last_link_id = nextLink;
fs.mkdirSync(path.dirname(destinationPath), { recursive: true });
fs.writeFileSync(destinationPath, `${JSON.stringify(workflow, null, 2)}\n`, "utf8");
console.log(
  `Wrote ${destinationPath} with selector node ${selector.id}, defaulting to T${String(tileNumber).padStart(3, "0")}.`,
);
