"""Deterministic detection of large smooth regions, measured from pixels only.

A run of adjacent cells or tiles that are all smooth AND share one color IS a
continuous surface by definition — no vision model guess is needed to know it
exists. This module only measures: where the smooth regions are, how big they
are, and what color they are. Naming the region (water, sky, field) stays with
the whole-image vision pass, which is *required* to identify every region
measured here. Nothing in this file describes content.
"""

import torch
import torch.nn.functional as F


# Whole-image analysis: a fixed working resolution keeps flatness thresholds
# scale-independent, and a 12x12 cell grid is fine enough to outline a region
# that covers only part of the frame.
WORKING_RESOLUTION = 144
CELL_GRID = 12
FLAT_CELL_STD = 0.03
FLAT_CELL_EDGE_FRACTION = 0.02
EDGE_STEP = 0.04
CELL_COLOR_MERGE_DISTANCE = 0.09
MIN_REGION_CELLS = 5  # ~3.5% of the image

# Tile grouping: adjacent uniform tiles this close in mean color are one surface.
TILE_COLOR_MERGE_DISTANCE = 0.08

NINE_LOCATION_LABELS = (
    "top left", "top center", "top right",
    "middle left", "center", "middle right",
    "bottom left", "bottom center", "bottom right",
)


def _axis_band(value):
    if value < 0.4:
        return 0
    if value < 0.6:
        return 1
    return 2


def point_location_label(x, y):
    """Map a normalized image point to one of the nine coarse location labels."""
    horizontal = ("left", "center", "right")[_axis_band(x)]
    vertical = ("top", "middle", "bottom")[_axis_band(y)]
    if vertical == "middle" and horizontal == "center":
        return "center"
    if vertical == "middle":
        return f"middle {horizontal}"
    return f"{vertical} {horizontal}"


def location_axes(value):
    """Parse any coarse location phrase into (horizontal, vertical) axes."""
    text = str(value or "").replace("-", " ").replace("_", " ").casefold()
    words = set(text.split())
    horizontal = "left" if "left" in words else "right" if "right" in words else "center"
    vertical = (
        "top"
        if words & {"top", "upper"}
        else "bottom"
        if words & {"bottom", "lower"}
        else "middle"
    )
    return horizontal, vertical


def color_distance(first, second):
    return sum((float(a) - float(b)) ** 2 for a, b in zip(first, second)) ** 0.5


def color_name(rgb):
    """Deterministic coarse color word for a measured mean color (0-1 RGB)."""
    r, g, b = (max(0.0, min(1.0, float(v))) for v in rgb)
    brightest = max(r, g, b)
    darkest = min(r, g, b)
    delta = brightest - darkest
    saturation = 0.0 if brightest <= 0.0 else delta / brightest
    if saturation < 0.12 or delta < 0.04:
        if brightest < 0.2:
            return "near-black"
        if brightest < 0.45:
            return "dark gray"
        if brightest < 0.75:
            return "gray"
        return "off-white"
    if brightest == r:
        hue = (60.0 * (((g - b) / delta) % 6.0))
    elif brightest == g:
        hue = 60.0 * ((b - r) / delta + 2.0)
    else:
        hue = 60.0 * ((r - g) / delta + 4.0)
    if 20.0 <= hue < 50.0 and brightest < 0.55:
        base = "brown"
    else:
        base = (
            "red" if hue < 20.0 or hue >= 340.0
            else "orange" if hue < 45.0
            else "yellow" if hue < 70.0
            else "green" if hue < 160.0
            else "teal" if hue < 200.0
            else "blue" if hue < 255.0
            else "purple" if hue < 290.0
            else "pink"
        )
    if brightest < 0.35:
        return f"dark {base}"
    if saturation < 0.35:
        return f"muted {base}"
    if brightest > 0.8 and saturation < 0.55:
        return f"light {base}"
    return base


def _working_image(image):
    tensor = image
    if tensor.ndim == 4:
        if int(tensor.shape[0]) != 1:
            raise ValueError(
                "Whole-image scene analysis accepts one source image at a time. "
                "Split image batches before Smart Upscaler so prompts cannot mix scenes."
            )
        tensor = tensor[0]
    tensor = tensor.detach().float().cpu()
    if tensor.shape[-1] > 3:
        tensor = tensor[..., :3]
    rgb = tensor.movedim(-1, 0).unsqueeze(0)
    return F.interpolate(
        rgb, size=(WORKING_RESOLUTION, WORKING_RESOLUTION), mode="area"
    )[0]


def flat_regions(image):
    """Measure large smooth single-color regions in the complete image.

    Returns a list of dicts sorted largest first:
    ``{"labels": [nine-location labels], "color_name": str,
       "mean_rgb": [r, g, b], "area_fraction": float}``
    """
    work = _working_image(image)
    # Measure all RGB channels. Simple channel averaging makes strong isoluminant
    # boundaries (for example red beside green) disappear and falsely classifies
    # them as one flat surface.
    edge = torch.zeros_like(work[0])
    horizontal_delta = (work[:, :, 1:] - work[:, :, :-1]).abs().amax(dim=0)
    vertical_delta = (work[:, 1:, :] - work[:, :-1, :]).abs().amax(dim=0)
    edge[:, 1:] = torch.maximum(edge[:, 1:], horizontal_delta)
    edge[1:, :] = torch.maximum(edge[1:, :], vertical_delta)

    cell = WORKING_RESOLUTION // CELL_GRID
    cells_edge = edge.reshape(CELL_GRID, cell, CELL_GRID, cell)
    cells_rgb = work.reshape(3, CELL_GRID, cell, CELL_GRID, cell)
    cell_std = cells_rgb.std(dim=(2, 4)).amax(dim=0)
    cell_edge_fraction = (cells_edge > EDGE_STEP).float().mean(dim=(1, 3))
    cell_rgb = cells_rgb.mean(dim=(2, 4))
    flat = (cell_std < FLAT_CELL_STD) & (cell_edge_fraction < FLAT_CELL_EDGE_FRACTION)

    visited = torch.zeros_like(flat)
    components = []
    for row in range(CELL_GRID):
        for column in range(CELL_GRID):
            if not bool(flat[row, column]) or bool(visited[row, column]):
                continue
            stack = [(row, column)]
            visited[row, column] = True
            member_cells = []
            while stack:
                cell_row, cell_column = stack.pop()
                member_cells.append((cell_row, cell_column))
                here = cell_rgb[:, cell_row, cell_column]
                for next_row, next_column in (
                    (cell_row - 1, cell_column),
                    (cell_row + 1, cell_column),
                    (cell_row, cell_column - 1),
                    (cell_row, cell_column + 1),
                ):
                    if not (0 <= next_row < CELL_GRID and 0 <= next_column < CELL_GRID):
                        continue
                    if bool(visited[next_row, next_column]) or not bool(
                        flat[next_row, next_column]
                    ):
                        continue
                    neighbor = cell_rgb[:, next_row, next_column]
                    if float((here - neighbor).pow(2).sum().sqrt()) > CELL_COLOR_MERGE_DISTANCE:
                        continue
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
            if len(member_cells) < MIN_REGION_CELLS:
                continue
            label_counts = {}
            red = green = blue = 0.0
            for cell_row, cell_column in member_cells:
                label = point_location_label(
                    (cell_column + 0.5) / CELL_GRID, (cell_row + 0.5) / CELL_GRID
                )
                label_counts[label] = label_counts.get(label, 0) + 1
                red += float(cell_rgb[0, cell_row, cell_column])
                green += float(cell_rgb[1, cell_row, cell_column])
                blue += float(cell_rgb[2, cell_row, cell_column])
            count = len(member_cells)
            mean_rgb = [red / count, green / count, blue / count]
            largest = max(label_counts.values())
            labels = [
                label
                for label in NINE_LOCATION_LABELS
                if label_counts.get(label, 0) >= max(2, largest // 6)
            ] or [max(label_counts, key=label_counts.get)]
            components.append(
                {
                    "labels": labels,
                    "color_name": color_name(mean_rgb),
                    "mean_rgb": [round(value, 4) for value in mean_rgb],
                    "area_fraction": round(count / (CELL_GRID * CELL_GRID), 4),
                }
            )
    components.sort(key=lambda item: item["area_fraction"], reverse=True)
    return components


def measured_regions_text(regions):
    """Render measured regions as an instruction block for the whole-image pass."""
    if not regions:
        return ""
    lines = []
    for index, region in enumerate(regions, start=1):
        lines.append(
            f"{index}. covering {' and '.join(region['labels'])}: flat single-color "
            f"({region['color_name']}), about {round(region['area_fraction'] * 100)}% of the image"
        )
    body = "\n".join(lines)
    return (
        "MEASURED FLAT REGIONS (from direct pixel measurement, trustworthy):\n"
        f"{body}\n"
        "Each measured entry above is a real large uniform surface. surface_map MUST contain an "
        "entry identifying every one of them: judge from the whole image what the thing actually "
        "IS (water, sky, sand, a lot, a field) and describe that thing itself. Never echo this "
        "measurement wording — never write \"flat region\", \"smooth region\", or \"blue area\" as "
        "the identity or description. Do not skip any measured entry and do not merge it into a "
        "structured area."
    )


def uncovered_region_problem(surface_map, regions):
    """Return a problem message when surface_map misses a measured flat region."""
    if not regions:
        return ""
    covered = set()
    if isinstance(surface_map, list):
        for entry in surface_map:
            if not isinstance(entry, dict):
                continue
            locations = entry.get("locations")
            if isinstance(locations, list):
                for location in locations:
                    covered.add(location_axes(location))
    for region in regions:
        region_axes = {location_axes(label) for label in region.get("labels", [])}
        if region_axes and not (region_axes & covered):
            place = " and ".join(region.get("labels", [])) or "the measured area"
            return f"surface_map does not identify the measured flat region at {place}"
    return ""


def tile_mean_rgb(tile_image):
    tensor = tile_image
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().float().cpu()
    if tensor.shape[-1] > 3:
        tensor = tensor[..., :3]
    return [float(value) for value in tensor.mean(dim=(0, 1))]


def uniform_tile_components(tiles):
    """Group grid-adjacent uniform tiles with matching mean color into surfaces.

    ``tiles`` is a list of dicts with ``tile_index``, ``row``, ``column``,
    ``evidence_class``, and ``mean_rgb``. Returns a list of components, each
    ``{"members": [uniform tile indexes], "attached": [adjacent same-color
    sparse tile indexes], "mean_rgb": [r, g, b]}``. A component needs at least
    two uniform members: a lone uniform tile is not a measured surface run.
    """
    by_position = {
        (
            int(t.get("source_index", 0)),
            int(t["row"]),
            int(t["column"]),
        ): t
        for t in tiles
    }
    parent = {}

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    uniform_tiles = [t for t in tiles if t.get("evidence_class") == "uniform"]
    for tile in uniform_tiles:
        parent[int(tile["tile_index"])] = int(tile["tile_index"])
    for tile in uniform_tiles:
        source = int(tile.get("source_index", 0))
        row, column = int(tile["row"]), int(tile["column"])
        for neighbor_position in ((row + 1, column), (row, column + 1)):
            neighbor = by_position.get((source, *neighbor_position))
            if neighbor is None or neighbor.get("evidence_class") != "uniform":
                continue
            if (
                color_distance(tile["mean_rgb"], neighbor["mean_rgb"])
                <= TILE_COLOR_MERGE_DISTANCE
            ):
                union(int(tile["tile_index"]), int(neighbor["tile_index"]))

    groups = {}
    for tile in uniform_tiles:
        groups.setdefault(find(int(tile["tile_index"])), []).append(tile)

    components = []
    for members in groups.values():
        if len(members) < 2:
            continue
        source_index = int(members[0].get("source_index", 0))
        member_indexes = {int(t["tile_index"]) for t in members}
        member_positions = {(int(t["row"]), int(t["column"])) for t in members}
        mean_rgb = [
            sum(float(t["mean_rgb"][channel]) for t in members) / len(members)
            for channel in range(3)
        ]
        attached = set()
        for tile in tiles:
            if tile.get("evidence_class") != "sparse":
                continue
            if int(tile.get("source_index", 0)) != source_index:
                continue
            row, column = int(tile["row"]), int(tile["column"])
            touches = any(
                position in member_positions
                for position in (
                    (row - 1, column), (row + 1, column),
                    (row, column - 1), (row, column + 1),
                )
            )
            if touches and color_distance(tile["mean_rgb"], mean_rgb) <= (
                TILE_COLOR_MERGE_DISTANCE * 1.5
            ):
                attached.add(int(tile["tile_index"]))
        components.append(
            {
                "source_index": source_index,
                "members": sorted(member_indexes),
                "attached": sorted(attached),
                "mean_rgb": [round(value, 4) for value in mean_rgb],
            }
        )
    components.sort(key=lambda item: len(item["members"]), reverse=True)
    return components
