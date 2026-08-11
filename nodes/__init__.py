from .audit import SmartTilePromptAuditLog
from .cache import SmartCachedTextGenerate, SmartCachedTilePromptGenerator
from .fidelity import SmartTileColorMatch
from .finalize import SmartTileFinalizer
from .processing import SmartSamplerTileSelector
from .review import SmartTileInspector
from .switching import SmartModelEngineSwitch
from .universal_prompting import SmartUnifiedPromptGuidance, SmartTileJobDirector
from .upscaled_tiling import SmartUpscaledTilePlanner

# The ten nodes of the Smart Upscaler pipeline, in pipeline order, plus the
# Generator Switch routing helper.
# Internal engines (SmartPromptGuidance, SmartTilePromptResolver, SmartTileSeed,
# SmartTileBlender, the base tile planners) stay importable from their modules but
# are not menu nodes.
NODE_CLASS_MAPPINGS = {
    "SmartUnifiedPromptGuidance": SmartUnifiedPromptGuidance,
    "SmartUpscaledTilePlanner": SmartUpscaledTilePlanner,
    "SmartCachedTextGenerate": SmartCachedTextGenerate,
    "SmartTileJobDirector": SmartTileJobDirector,
    "SmartCachedTilePromptGenerator": SmartCachedTilePromptGenerator,
    "SmartSamplerTileSelector": SmartSamplerTileSelector,
    "SmartTileColorMatch": SmartTileColorMatch,
    "SmartTileFinalizer": SmartTileFinalizer,
    "SmartTileInspector": SmartTileInspector,
    "SmartTilePromptAuditLog": SmartTilePromptAuditLog,
    "SmartModelEngineSwitch": SmartModelEngineSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartUnifiedPromptGuidance": "Prompt Director: Editable Master Instructions",
    "SmartUpscaledTilePlanner": "Output-Scale Tiles: AI or Standard Resize",
    "SmartCachedTextGenerate": "Automatic Whole-Image Analysis + Cache",
    "SmartTileJobDirector": "Tile Job Director: Exact Local Tiles",
    "SmartCachedTilePromptGenerator": "Automatic Exact-Tile Prompt + Cache",
    "SmartSamplerTileSelector": "Sampler Tile Test Selector (Optional)",
    "SmartTileColorMatch": "Color Match to Original (Optional)",
    "SmartTileFinalizer": "Stitch Tiles Into One Picture",
    "SmartTileInspector": "Smart Tile Prompt Inspector",
    "SmartTilePromptAuditLog": "All Prompts Viewer + Log",
    "SmartModelEngineSwitch": "Generator Switch (only selected runs)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
