"""Generator Switch: one dropdown that routes between image generator chains.

The whole point over group muting or third-party switches is laziness: only
the selected generator input is ever evaluated, so the nine other sampler
chains cost nothing - no model loads, no VRAM, no time. Any signal type passes
through (IMAGE, MODEL, CONDITIONING, INT, ...), so the same node also covers
small value switches.

"Generator" replaced "engine" in every visible name. The class name is the
registered node type and is deliberately unchanged: every saved graph refers to
this node by that string, so renaming it would break workflows already in the
wild. Legacy `engine_*` socket names and "Engine 3" dropdown values are still
accepted here, and the web extension renames the sockets on load.
"""

import re


MAX_GENERATORS = 10


class _AnyType(str):
    """Wildcard socket type: compares equal to every other type."""

    def __ne__(self, _other):
        return False

    def __eq__(self, _other):
        return True

    __hash__ = str.__hash__


ANY_TYPE = _AnyType("*")

GENERATOR_CHOICES = tuple(f"Generator {index}" for index in range(1, MAX_GENERATORS + 1))


def _slot_name(kwargs, index):
    """The socket name this graph actually uses for a slot.

    A graph saved before the rename sends `engine_3`; a current one sends
    `generator_3`. check_lazy_status must name a socket that exists, or the
    lazy fetch asks for something the prompt has never heard of.
    """
    legacy = f"engine_{index}"
    current = f"generator_{index}"
    if legacy in kwargs and current not in kwargs:
        return legacy
    return current


def _selected_generator(generator, generator_number=None):
    """The 1-based selected slot from the dropdown or the override number.

    The web UI relabels dropdown entries with the connected chain's name
    ("3: Flux 1"), so the stored value is parsed for its first number. Legacy
    "Engine 3" values parse the same way, and unknown strings fall back to slot
    1 instead of failing the run.
    """
    if generator_number is not None:
        try:
            return min(max(int(generator_number), 1), MAX_GENERATORS)
        except (TypeError, ValueError):
            pass
    match = re.search(r"\d+", str(generator or ""))
    if match:
        return min(max(int(match.group(0)), 1), MAX_GENERATORS)
    return 1


class SmartModelEngineSwitch:
    """Dropdown switch between up to ten generator chains; only the chosen one runs."""

    CATEGORY = "Smart Upscaler/Routing"
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("selected",)
    FUNCTION = "route"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"generator_{index}": (
                ANY_TYPE,
                {
                    "lazy": True,
                    "tooltip": f"Generator chain {index}. Only the selected generator is executed.",
                },
            )
            for index in range(1, MAX_GENERATORS + 1)
        }
        optional["generator_number"] = (
            "INT",
            {
                "forceInput": True,
                "default": 0,
                "min": 0,
                "max": MAX_GENERATORS,
                "tooltip": "Optional override: connect one shared number to keep several switches in sync. 0 or unconnected = use the dropdown.",
            },
        )
        return {
            "required": {
                "generator": (
                    list(GENERATOR_CHOICES),
                    {
                        "default": GENERATOR_CHOICES[0],
                        "label": "Generator",
                        "tooltip": "Which connected generator chain to run. Chains that are not selected are never executed - no model loads, no VRAM, no time.",
                    },
                ),
            },
            "optional": optional,
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **_kwargs):
        # The dropdown stores relabeled values ("3: Flux 1") and the wildcard
        # sockets carry any type; both would fail stock validation.
        return True

    def check_lazy_status(self, generator=None, generator_number=None, **kwargs):
        selected = _selected_generator(
            generator if generator is not None else kwargs.get("engine"),
            generator_number or kwargs.get("engine_number") or None,
        )
        name = _slot_name(kwargs, selected)
        if kwargs.get(name) is None:
            return [name]
        return []

    def route(self, generator=None, generator_number=None, **kwargs):
        selected = _selected_generator(
            generator if generator is not None else kwargs.get("engine"),
            generator_number or kwargs.get("engine_number") or None,
        )
        value = kwargs.get(f"generator_{selected}")
        if value is None:
            value = kwargs.get(f"engine_{selected}")
        if value is None:
            connected = sorted(
                {
                    int(key.rsplit("_", 1)[1])
                    for key, item in kwargs.items()
                    if item is not None
                    and key.rsplit("_", 1)[-1].isdigit()
                    and key.startswith(("generator_", "engine_"))
                }
            )
            listed = ", ".join(str(index) for index in connected) or "none"
            raise ValueError(
                f"Generator Switch: generator {selected} is selected but not "
                f"connected (connected generators: {listed}). Pick a connected "
                "generator in the dropdown."
            )
        return (value,)
