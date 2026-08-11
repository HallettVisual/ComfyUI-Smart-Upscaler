import json
import re


def _load_tile_metadata(tile_metadata_json):
    try:
        metadata = json.loads(tile_metadata_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Tile metadata must be valid JSON from Smart Tile Planner.") from exc

    if not isinstance(metadata, dict) or not isinstance(metadata.get("tiles"), list):
        raise ValueError("Tile metadata is missing its tiles list.")

    required_metadata = ("image_width", "image_height", "tile_width", "tile_height")
    missing_metadata = [key for key in required_metadata if key not in metadata]
    if missing_metadata:
        raise ValueError(f"Tile metadata is missing: {', '.join(missing_metadata)}.")

    required_tile = ("tile_index", "source_index", "row", "column", "position", "x", "y", "width", "height")
    seen_indexes = set()
    for tile in metadata["tiles"]:
        if not isinstance(tile, dict):
            raise ValueError("Every tile metadata entry must be an object.")
        missing_tile = [key for key in required_tile if key not in tile]
        if missing_tile:
            raise ValueError(f"A tile metadata entry is missing: {', '.join(missing_tile)}.")
        tile_index = tile["tile_index"]
        if tile_index in seen_indexes:
            raise ValueError(f"Tile metadata contains duplicate tile index {tile_index}.")
        seen_indexes.add(tile_index)

    return metadata


def _clean_generated_text(value):
    text = str(value).strip()
    while text.endswith("<end_of_turn>"):
        text = text[: -len("<end_of_turn>")].rstrip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    return text


def _loads_tolerant(value):
    """Parse a JSON object from model text, surviving prose wrappers and truncation.

    A verbose or length-capped VLM often returns a JSON object that is wrapped in a code
    fence, preceded by a sentence, or cut off before its closing braces. Rather than abort
    the whole run, extract the outermost object and, if it was truncated, close any open
    strings/brackets so the completed portion is still usable. Returns a dict or None.
    """
    text = _clean_generated_text(value)
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None
    text = text[start:]

    depth_stack = []
    in_string = False
    escaped = False
    balanced_end = None
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth_stack.append(char)
        elif char in "}]":
            if depth_stack:
                depth_stack.pop()
            if not depth_stack:
                balanced_end = index + 1
                break

    if balanced_end is not None:
        try:
            parsed = json.loads(text[:balanced_end])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    # Truncated mid-object: close the dangling string and any open brackets.
    repaired = text
    if in_string:
        repaired += '"'
    repaired = re.sub(r"[,:]\s*$", "", repaired.rstrip())
    for opener in reversed(depth_stack):
        repaired += "}" if opener == "{" else "]"
    try:
        parsed = json.loads(repaired)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None



_INVALID_VISIBLE_CAPTION = re.compile(
    r"\b(?:no visible content|nothing visible|masked placeholder|only black background|"
    r"no content to describe|empty (?:image|crop|tile))\b",
    re.IGNORECASE,
)

_INVALID_TARGET_PLACEHOLDER = re.compile(
    r"\b(?:desired finished tile|desired-result prompt|concise desired repaired result|"
    r"only content visibly present|repairs? required (?:for|in) (?:that|this) (?:content|tile)|"
    r"target prompt goes here)\b",
    re.IGNORECASE,
)

_AMBIGUOUS_BACKGROUND_ONLY = re.compile(
    r"^\s*(?:(?:a|an|the)\s+)?(?:[a-z0-9-]+\s+){0,8}background"
    r"(?:\s+with\s+[^.;]+)?[.!]?\s*$",
    re.IGNORECASE,
)

_VIEWPOINT_DESCRIPTION = re.compile(
    r"\b(?:aerial|overhead|top-down|high-angle|low-angle|street-level|ground-level|"
    r"oblique|bird(?:'s|-)?eye|close-up|macro)\b",
    re.IGNORECASE,
)

_CAMERA_VIEW_TOKEN = re.compile(
    r"\b(?:aerial|overhead|top-down|high-angle|low-angle|street-level|ground-level|"
    r"oblique|bird(?:'s|-)?eye|close-up|macro)(?:\s+(?:view|perspective|shot))?\b|"
    r"\blooking\s+(?:down|up|across|toward(?:\s+the)?\s+camera)\b",
    re.IGNORECASE,
)

_GENERIC_REGION_PLACEHOLDER = re.compile(
    r"\b(?:continuous visible region|continuous source region|"
    r"locally uniform continuous region|one continuous source region)\b",
    re.IGNORECASE,
)


def _safe_view_context(value):
    """Defensively strip object inventory from inherited whole-image view text."""
    text = " ".join(str(value or "").split())
    matches = [match.group(0).strip() for match in _CAMERA_VIEW_TOKEN.finditer(text)]
    result = []
    for item in matches:
        if item.casefold() not in {existing.casefold() for existing in result}:
            result.append(item)
    return ", ".join(result)


_STANDALONE_IMAGE_DESCRIPTION = re.compile(
    r"^(?:(?:a|an)\s+)?"
    r"(?:[a-z][a-z-]*\s+){0,9}"
    r"(?:image|photo(?:graph)?|rendering|view)\s+(?:of|showing|depicting)\s+",
    re.IGNORECASE,
)


def _target_detail(value, preserve_viewpoint=False):
    """Remove text-to-image framing before combining local detail with an edit command."""
    detail = _strip_positive_absence_clauses(value).strip().rstrip(" .")
    if not (preserve_viewpoint and _VIEWPOINT_DESCRIPTION.search(detail)):
        detail = _STANDALONE_IMAGE_DESCRIPTION.sub("", detail).strip()
    if detail and detail[0].islower():
        detail = detail[0].upper() + detail[1:]
    return detail


def _extract_json_string_field(text, key):
    match = re.search(
        rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)(?:"|$)',
        text,
        re.DOTALL,
    )
    if not match:
        return ""
    encoded = f'"{match.group(1)}"'
    try:
        return str(json.loads(encoded)).strip()
    except json.JSONDecodeError:
        return match.group(1).replace(r'\"', '"').strip()


def _source_caption_payload(value):
    response = _clean_generated_text(value)
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    result = {}
    for key in (
        "local_caption",
        "dominant_region",
        "visible_boundaries",
        "surface_id",
        "surface_prompt",
        "object_id",
        "object_prompt",
        "local_features",
        "corrections_applied",
        "required_change",
        "target_prompt",
    ):
        field = payload.get(key)
        if key == "local_caption" and field is None:
            field = payload.get("positive_prompt") or payload.get("prompt")
        if field is None:
            field = _extract_json_string_field(response, key)
        if isinstance(field, list):
            field = "; ".join(str(item).strip() for item in field if str(item).strip())
        result[key] = str(field or "").strip()
    if not result["corrections_applied"]:
        result["corrections_applied"] = result["required_change"]
    result["raw_response"] = response
    return result


def _normalized_excluded_terms(value):
    if isinstance(value, (list, tuple)):
        candidates = value
    else:
        candidates = re.split(r"[,;\n]+", str(value or ""))
    result = []
    for candidate in candidates:
        term = re.sub(r"^\s*(?:no|not)\s+", "", str(candidate), flags=re.IGNORECASE)
        term = term.strip().strip(".\"'")
        if term and term.casefold() not in {item.casefold() for item in result}:
            result.append(term)
    return result


def _false_detection_forms(term):
    """The written forms of one banned term, so plurals cannot slip past.

    A user who bans "wires" means wire. The old exact match let every singular
    through: the wet-hair portrait banned `wires` and still shipped "a thin,
    translucent wire" to the sampler. Only simple regular English endings are
    generated - no synonyms and no stemming, so banning one word can never
    silently remove a different one.
    """
    word = str(term or "").strip()
    if not word:
        return []
    forms = {word}
    lowered = word.casefold()
    if len(lowered) > 3 and lowered.endswith("ies"):
        forms.add(word[:-3] + "y")
    elif len(lowered) > 3 and lowered.endswith("es") and lowered[-3] in "sxzho":
        forms.add(word[:-2])
    elif len(lowered) > 3 and lowered.endswith("s") and not lowered.endswith("ss"):
        forms.add(word[:-1])
    elif lowered.endswith(("s", "x", "z", "ch", "sh")):
        forms.add(word + "es")
    elif lowered.endswith("y") and len(lowered) > 2 and lowered[-2] not in "aeiou":
        forms.add(word[:-1] + "ies")
    else:
        forms.add(word + "s")
    # Longest first so "wires" is consumed before "wire" can match half of it.
    return sorted(forms, key=len, reverse=True)


def _excluded_detection_problem(value, excluded_terms):
    text = str(value or "")
    for term in _normalized_excluded_terms(excluded_terms):
        for form in _false_detection_forms(term):
            if re.search(rf"(?<!\w){re.escape(form)}(?!\w)", text, re.IGNORECASE):
                return f'user-confirmed false detection "{term}"'
    return ""


def _apply_known_false_detection_guard(value, excluded_terms=()):
    """Remove user-confirmed false concepts after the VLM repeats them on retry."""
    response = _clean_generated_text(value)
    # Use the same tolerant, normalized field view as validation. This handles
    # complete JSON, truncated JSON, and list-valued fields identically instead
    # of finding an excluded term in a flattened list that cleanup cannot edit.
    extracted = _source_caption_payload(response)
    payload = {
        "local_caption": extracted["local_caption"],
        "dominant_region": extracted["dominant_region"],
        "visible_boundaries": extracted["visible_boundaries"],
        "surface_id": extracted["surface_id"],
        "surface_prompt": extracted["surface_prompt"],
        "object_id": extracted["object_id"],
        "object_prompt": extracted["object_prompt"],
        "local_features": extracted["local_features"],
        "corrections_applied": extracted["corrections_applied"],
        "target_prompt": extracted["target_prompt"],
    }
    terms = _normalized_excluded_terms(excluded_terms)
    if not terms:
        return value

    def contains_excluded(text):
        return bool(_excluded_detection_problem(text, terms))

    def clean_field(field):
        text = str(field or "").strip()
        if not text or not contains_excluded(text):
            return text
        # Remove only clauses that contain a confirmed false concept. Other
        # visible content remains intact and no replacement object is invented.
        clauses = re.split(
            r"\s*(?:[,;.]|\b(?:and|with|beside|alongside|near|next\s+to)\b)\s*",
            text,
            flags=re.IGNORECASE,
        )
        kept = [clause.strip() for clause in clauses if clause.strip() and not contains_excluded(clause)]
        return ", ".join(kept).strip(" ,;:-")

    for key in (
        "local_caption",
        "dominant_region",
        "visible_boundaries",
        "surface_id",
        "surface_prompt",
        "object_id",
        "object_prompt",
        "local_features",
        "corrections_applied",
        "required_change",
        "target_prompt",
    ):
        payload[key] = clean_field(payload.get(key, ""))
    # Canonical id handles (e.g. "pier_1") embed the concept without a word boundary, so
    # clear an id outright when a false term appears anywhere in it. This stops a scrubbed
    # entry from being re-selected by its id.
    for id_key in ("surface_id", "object_id"):
        value = str(payload.get(id_key, ""))
        if any(term.casefold() in value.casefold() for term in terms):
            payload[id_key] = ""
    local_caption = str(payload.get("local_caption") or "").strip()
    if local_caption:
        if not str(payload.get("dominant_region") or "").strip():
            payload["dominant_region"] = local_caption
        # The deterministic edit action is prepended later. If removal of a
        # confirmed false concept empties the target, literal local evidence is
        # the safest non-inventive target detail.
        if not str(payload.get("target_prompt") or "").strip():
            payload["target_prompt"] = local_caption
    return json.dumps(payload, ensure_ascii=False)


def _apply_missing_local_caption_guard(value):
    """Recover an omitted local_caption from other exact-tile fields after retry."""
    payload = _source_caption_payload(value)
    if payload["local_caption"]:
        return value
    fallback = payload["dominant_region"]
    if not fallback and payload["target_prompt"]:
        fallback = _target_detail(payload["target_prompt"], preserve_viewpoint=True)
    if not fallback:
        return value
    repaired = {
        "local_caption": fallback,
        "dominant_region": payload["dominant_region"] or fallback,
        "visible_boundaries": payload["visible_boundaries"],
        "surface_id": payload["surface_id"],
        "surface_prompt": payload["surface_prompt"],
        "local_features": payload["local_features"],
        "corrections_applied": payload["corrections_applied"],
        "target_prompt": payload["target_prompt"],
    }
    return json.dumps(repaired, ensure_ascii=False)


def _apply_plain_local_caption_guard(value):
    """Convert a focused plain-language tile inspection into the validated schema."""
    payload = _source_caption_payload(value)
    fallback = payload["local_caption"] or payload["dominant_region"]
    if not fallback and payload["target_prompt"]:
        fallback = _target_detail(payload["target_prompt"], preserve_viewpoint=True)
    if not fallback:
        response = payload["raw_response"].strip()
        if response and not response.lstrip().startswith(("{", "[")):
            fallback = response
    fallback = re.sub(
        r"^\s*(?:source|caption|tile|description)\s*:\s*",
        "",
        str(fallback or ""),
        flags=re.IGNORECASE,
    ).strip().strip('`"').strip()
    if not fallback:
        return value
    repaired = {
        "local_caption": fallback,
        "dominant_region": fallback,
        "visible_boundaries": "",
        "surface_id": "",
        "surface_prompt": "",
        "local_features": "",
        "corrections_applied": "",
        # The deterministic edit action is added separately by the resolver.
        # Reusing literal local evidence here is the safest target when the
        # caption model could not satisfy the richer JSON contract twice.
        "target_prompt": fallback,
    }
    return json.dumps(repaired, ensure_ascii=False)


def _source_caption_problem(
    value,
    evidence_class="structured",
    require_target=False,
    excluded_terms=(),
    visual_complexity="",
    require_concrete_uniform_region=False,
):
    payload = _source_caption_payload(value)
    local_caption = payload["local_caption"]
    if not local_caption:
        return "missing local_caption"
    if _INVALID_VISIBLE_CAPTION.search(local_caption):
        return "caption incorrectly reports an empty or placeholder tile"
    is_uniform = str(evidence_class).lower() == "uniform"
    if _AMBIGUOUS_BACKGROUND_ONLY.fullmatch(local_caption) and (
        not is_uniform or require_concrete_uniform_region
    ):
        return 'caption calls the visible region a "background" instead of identifying its surface or region'
    if (
        is_uniform
        and require_concrete_uniform_region
        and _GENERIC_REGION_PLACEHOLDER.search(local_caption)
    ):
        return "uniform tile uses a generic region placeholder instead of naming the visible surface"
    if require_target:
        target_prompt = payload["target_prompt"]
        if not target_prompt:
            return "missing target_prompt"
        if _AMBIGUOUS_BACKGROUND_ONLY.fullmatch(target_prompt) and (
            not is_uniform or require_concrete_uniform_region
        ):
            return 'target_prompt calls the visible region a "background" instead of identifying its surface or region'
        if (
            is_uniform
            and require_concrete_uniform_region
            and _GENERIC_REGION_PLACEHOLDER.search(target_prompt)
        ):
            return "uniform tile target uses a generic region placeholder instead of naming the visible surface"
        if _INVALID_TARGET_PLACEHOLDER.search(local_caption) or _INVALID_TARGET_PLACEHOLDER.search(
            target_prompt
        ):
            return "model repeated the output template instead of analyzing the tile"
    checked_fields = " ".join(
        payload[key]
        for key in (
            "local_caption",
            "dominant_region",
            "visible_boundaries",
            "surface_prompt",
            "local_features",
            "corrections_applied",
            "target_prompt",
        )
        if payload[key]
    )
    excluded_problem = _excluded_detection_problem(checked_fields, excluded_terms)
    if excluded_problem:
        return f"caption contains {excluded_problem}"
    return ""


_ABSENCE_PREFIX = re.compile(
    r"^(?:no\b|without\b|absence\s+of\b|free\s+of\b|lacks?\b|"
    r"does\s+not\s+contain\b|do\s+not\s+include\b)",
    re.IGNORECASE,
)


def _strip_positive_absence_clauses(value):
    """Keep absent concepts out of positive diffusion conditioning."""
    text = str(value).strip()
    if not text:
        return ""
    original = text
    # Models often embed an absence after useful positive content, for example
    # "dark foliage with no visible light sources, preserving texture". Remove
    # that embedded absence while keeping the positive clauses on either side.
    text = re.sub(
        r"\b(?:with|and)\s+(?:no\b|without\b|an?\s+absence\s+of\b)[^,;.!]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    clauses = re.split(
        r"(?<=[.!;])\s+|,\s+(?=(?:no\b|without\b|absence\s+of\b|free\s+of\b|"
        r"lacks?\b|does\s+not\s+contain\b|do\s+not\s+include\b))|"
        r"[—–-]\s+(?=(?:no\b|without\b|absence\s+of\b|free\s+of\b))",
        text,
        flags=re.IGNORECASE,
    )
    present = [clause for clause in clauses if clause.strip(" ,;.")]
    kept = [clause.strip(" ,;.") for clause in clauses if not _ABSENCE_PREFIX.match(clause.strip())]
    kept = [clause for clause in kept if clause]
    # Nothing was actually removed: hand back the original wording. Rejoining
    # clauses with commas turns the model's sentences into a comma run
    # ("... across the frame, Sunlight filters ..."), which reads as a tag list
    # to a text encoder. Guards must subtract, never reflow clean text.
    if text == original and len(kept) == len(present):
        return original
    return ", ".join(kept)


_ARTIFACT_DAMAGE = re.compile(
    r"\b(?:debris|rubble|wreckage|broken|damaged?|cracked|crumbling|ruined|dilapidated|"
    r"collapsed?|deteriorat\w*|decaying|derelict|torn|melted|melting|smeared|warped|"
    r"distorted|garbled|corrupted|glitch\w*|malformed|mangled)\b",
    re.IGNORECASE,
)

# Softer degradation words. In a repair task ("Google Image Enhance") these still
# describe the defect instead of the wanted result ("minor structural wear"), so the
# aggressive pass removes them too. Other tasks keep them: a genuinely weathered barn
# in a faithful upscale is real content.
_ARTIFACT_WEAR = re.compile(
    r"\b(?:wear|worn|weather-?worn|weathered|erod\w*|patchy|stained|grimy|dingy|"
    r"faded|peeling|rusty|rusted)\b",
    re.IGNORECASE,
)

# How the CAPTURE looks, not what the picture contains. A tile that reports "the
# image has a low-resolution, pixelated quality" hands the sampler an instruction
# to render low-resolution pixelation. Swept only for repair tasks: a faithful
# upscale is explicitly told that blur is real content ("out-of-focus areas stay
# just as soft"), so these words must survive there.
_CAPTURE_QUALITY = re.compile(
    r"\b(?:pixelated|pixellated|pixelation|pixellation|low[-\s]?res|low[-\s]?resolution|"
    r"lowres|blurry|blurred|blurriness|grainy|graininess|fuzzy|jagged|aliased|"
    r"upsampled|upscaled|interpolated|compress\w*|jpe?g)\b",
    re.IGNORECASE,
)


def _merge_canonical_clause(lead, addition):
    """Append canonical wording without repeating what the tile already said.

    The canonical phrase is usually several clauses. A whole-string containment
    test let the entire phrase through whenever only one of its clauses was
    already present, so tile prompts carried "glass-clad high-rise buildings
    with reflective surfaces, sharp edges, and clean lines" twice over.
    """
    lead_text = str(lead or "").strip()
    addition_text = str(addition or "").strip()
    if not addition_text:
        return lead_text
    if not lead_text:
        return addition_text
    existing = {
        clause.strip(" .,;").casefold()
        for clause in re.split(r"\s*;\s*", lead_text)
        if clause.strip()
    }
    kept = [
        clause.strip(" .,;")
        for clause in re.split(r"\s*;\s*", addition_text)
        if clause.strip() and clause.strip(" .,;").casefold() not in existing
    ]
    return "; ".join([lead_text] + kept) if kept else lead_text


def _repair_artifact_language(value, keep_terms=(), aggressive=False):
    """Remove reconstruction-artifact words from a positive target.

    In a Google Earth / photogrammetry capture, "broken", "debris", "melted", "distorted"
    and similar are capture artifacts to repair, never real content — so they must not
    reach the sampler as positive nouns. The prompt must say what SHOULD be there, not
    what is there. A clause is dropped only when the artifact word is its content; a
    clause is kept if the user explicitly asked for that word. ``aggressive`` extends
    the sweep to soft wear/degradation words for repair tasks.
    """
    text = str(value or "").strip()
    if not text:
        return text
    if not _ARTIFACT_DAMAGE.search(text) and not (
        aggressive
        and (_ARTIFACT_WEAR.search(text) or _CAPTURE_QUALITY.search(text))
    ):
        return text
    keep = " ".join(str(term) for term in keep_terms).lower()
    clauses = re.split(
        r"\s*(?:[,;.]|\b(?:and|with|including|featuring|plus)\b)\s*",
        text,
        flags=re.IGNORECASE,
    )
    kept = []
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        match = _ARTIFACT_DAMAGE.search(clause) or (
            (_ARTIFACT_WEAR.search(clause) or _CAPTURE_QUALITY.search(clause))
            if aggressive
            else None
        )
        if match:
            if match.group(0).lower() in keep:
                kept.append(clause)
            # Otherwise drop the clause: it describes an artifact to repair, not content.
            continue
        kept.append(clause)
    return ", ".join(kept).strip(" ,;:-")


_ATMOSPHERE_TERM = re.compile(
    r"\b(?:sky|skies|sunny|overcast|cloud\w*|sunlight|sunshine|sunset|sunrise|"
    r"dusk|dawn|hazy|foggy|misty)\b",
    re.IGNORECASE,
)


def _atmosphere_family(value):
    token = str(value or "").casefold()
    if token in ("sky", "skies"):
        return "sky"
    for prefix in ("cloud", "sun", "haze", "fog", "mist"):
        if token.startswith(prefix):
            return prefix
    return token


def _strip_unsupported_atmosphere(value, evidence_text="", keep_terms=()):
    """Drop sky/weather framing the tile's own pixels do not support.

    A tile model likes to close with flourishes such as "all under a clear sky" even
    for an aerial crop that contains no sky at all — and the sampler will then paint
    one. A clause mentioning sky or weather survives only when the tile's own local
    caption also mentions it, or the user explicitly asked for it.
    """
    text = str(value or "").strip()
    if not text or not _ATMOSPHERE_TERM.search(text):
        return text
    supported_text = f"{evidence_text} {' '.join(str(term) for term in keep_terms)}"
    supported = {
        _atmosphere_family(match.group(0))
        for match in _ATMOSPHERE_TERM.finditer(supported_text)
    }
    clauses = re.split(
        r"\s*(?:[,;.]|\b(?:and|under|beneath|below|against)\b)\s*",
        text,
        flags=re.IGNORECASE,
    )
    kept = []
    dropped = False
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        match = _ATMOSPHERE_TERM.search(clause)
        if match and _atmosphere_family(match.group(0)) not in supported:
            dropped = True
            continue
        kept.append(clause)
    # Every atmosphere term was supported, so nothing needed removing: keep the
    # model's own sentences and conjunctions instead of rebuilding a comma run.
    if not dropped:
        return text
    return ", ".join(kept).strip(" ,;:-")


_TILE_POSITION_LABELS = (
    "top left",
    "top center",
    "top right",
    "middle left",
    "middle right",
    "bottom left",
    "bottom center",
    "bottom right",
)


def _strip_own_tile_location(value, position):
    """Drop the tile's own whole-image location when echoed as an in-tile place.

    A bottom-center tile that writes "a mossy rock edge at bottom center" is
    repeating where the CROP sits in the full image, not where the rock sits in
    the crop - and a denoise sampler reads it as a placement instruction. The
    label is only removed when it is exactly this tile's own whole-image
    position, where the phrase can carry no information the sampler can use; a
    genuinely in-tile position for any other area survives untouched. The
    single-word "center" tile is left alone because "in the center" is far more
    often a real in-tile statement.
    """
    text = str(value or "").strip()
    label = str(position or "").strip().casefold()
    if not text or label not in _TILE_POSITION_LABELS:
        return text
    vertical, horizontal = label.split()
    words = rf"{re.escape(vertical)}\s+{re.escape(horizontal)}"
    frame = r"(?:image|frame|tile|picture|photo|crop)"
    cleaned = re.sub(
        rf"(?<!\w)(?:at|in|on|across|near|toward|towards|along|from)\s+(?:the\s+)?{words}"
        rf"(?:\s+(?:of|in)\s+(?:the\s+)?{frame})?(?!\w)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        rf"(?<!\w)(?:the\s+)?{words}\s+(?:of|in)\s+(?:the\s+)?{frame}\s*[:,-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    if cleaned == text:
        return text
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,;.])", r"\1", cleaned)
    cleaned = re.sub(r"([,;])(?:\s*[,;])+", r"\1", cleaned)
    cleaned = cleaned.strip(" ,;:-")
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned or text


_FRAME_TALK = re.compile(r"\b(?:tile|crop|frame|image|picture|photo)\b", re.IGNORECASE)

# Words that describe the CROP rather than what is in it. A clause built only
# from these says nothing a sampler can paint.
_FRAME_TALK_STRUCTURAL = {
    "a", "an", "and", "are", "area", "areas", "across", "at", "boundaries",
    "boundary", "center", "centre", "clipped", "content", "corner", "corners",
    "crop", "cut", "cuts", "cutting", "distinct", "edge", "edges", "exact",
    "feature", "features", "frame", "image", "in", "is", "it", "its", "left",
    "middle", "no", "not", "object", "objects", "of", "on", "only", "or",
    "partial", "partially", "photo", "picture", "pixels", "region", "regions",
    "remain", "remaining", "remains", "right", "same", "sharp", "side", "sides",
    "similar", "that", "the", "this", "through", "tile", "to", "top", "visible",
    "was", "were", "with", "within", "bottom",
}


def _strip_frame_talk(value):
    """Drop clauses that describe the crop instead of its contents.

    A denoise sampler renders every noun it is given. "edge-cut features remain
    clipped at the same image edge" is a rule for an edit model, but read as
    content it asks for thin cut fragments - which is how stick-like debris
    appears in smooth areas. A clause is only dropped when EVERY word in it is
    structural, so real content ("the right edge cuts through a rocky outcrop")
    survives untouched.
    """
    text = str(value or "").strip()
    if not text or not _FRAME_TALK.search(text):
        return text
    clauses = re.split(r"\s*[;,.]\s*", text)
    kept = []
    dropped = False
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        words = re.findall(r"[a-z]+", clause.casefold())
        if (
            words
            and _FRAME_TALK.search(clause)
            and all(word in _FRAME_TALK_STRUCTURAL for word in words)
        ):
            dropped = True
            continue
        kept.append(clause)
    if not dropped:
        return text
    return ", ".join(kept).strip(" ,;:-")


_COLOR_FAMILIES = {
    "red": ("red", "crimson", "scarlet", "maroon", "burgundy"),
    "orange": ("orange", "amber", "ochre"),
    "yellow": ("yellow", "golden"),
    "green": ("green", "olive", "emerald", "lime", "mint"),
    "blue": ("blue", "navy", "azure", "cobalt"),
    "purple": ("purple", "violet", "indigo", "lavender", "mauve"),
    "pink": ("pink", "magenta", "fuchsia", "rose"),
    "cyan": ("cyan", "teal", "turquoise", "aqua"),
    "brown": ("brown", "beige", "tan", "khaki"),
    "black": ("black",),
    "white": ("white", "cream", "ivory"),
    "gray": ("gray", "grey"),
    "peach": ("peach", "coral"),
}
_COLOR_MODIFIERS = (
    "very",
    "dark",
    "light",
    "pale",
    "deep",
    "bright",
    "vivid",
    "muted",
    "warm",
    "cool",
    "rich",
    "soft",
)


def _strip_color_words(value, keep_sources=()):
    """Remove sampler-facing color language except colors the user requested."""
    text = str(value or "").strip()
    if not text:
        return text
    keep_text = " ".join(str(source or "") for source in keep_sources)
    kept_families = {
        family
        for family, terms in _COLOR_FAMILIES.items()
        if any(
            re.search(rf"(?<!\w){re.escape(term)}(?!\w)", keep_text, re.IGNORECASE)
            for term in terms
        )
    }
    removable_terms = sorted(
        {
            term
            for family, terms in _COLOR_FAMILIES.items()
            if family not in kept_families
            for term in terms
        },
        key=len,
        reverse=True,
    )
    if not removable_terms:
        return text
    terms = "|".join(re.escape(term) for term in removable_terms)
    modifiers = "|".join(_COLOR_MODIFIERS)
    color_phrase = re.compile(
        rf"(?<!\w)(?:(?:(?:{modifiers})\s+)*)"
        rf"(?:{terms})(?:\s*[-/]\s*(?:{terms}))?"
        rf"(?:[-\s]colou?red)?(?!\w)",
        re.IGNORECASE,
    )
    cleaned = color_phrase.sub(" ", text)
    cleaned = re.sub(r"\b(?:colorful|colourful|multicolou?red)\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*-\s*(?=[,;.]|$)", " ", cleaned)
    cleaned = re.sub(r"\b(?:with|in|and)\s+(?=[,;.]|$)", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+([,;.])", r"\1", cleaned)
    cleaned = re.sub(r"([,;])(?:\s*[,;])+", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,;:.-")


_SURFACE_PROCESS_WORD = re.compile(
    r"\b(?:region|area|background|zone)\b", re.IGNORECASE
)

# A canonical surface phrase is stamped onto tiles that contain ONLY that surface.
# Any discrete object it mentions — "reflections of surrounding buildings" — will be
# painted into every one of those tiles. So a surface description must be object-free.
_SURFACE_DISCRETE_OBJECT = re.compile(
    r"\b(?:surrounding|build\w*|tower\w*|skyscraper\w*|boat\w*|ship\w*|ferr(?:y|ies)|"
    r"dock\w*|pier\w*|marina\w*|car(?:s)?\b|vehicle\w*|truck\w*|tree\w*|forest\w*|"
    r"person|people|house\w*|bridge\w*|road\w*|street\w*|window\w*|roof\w*)\b",
    re.IGNORECASE,
)


# An ambiguous tile guesses its identity from color: blue mist reads as water,
# pale haze reads as sky. The validated whole-image brief knows WHERE these
# families actually are, so a uniform/sparse tile may only claim one when a
# location-matched canonical candidate of that family reached it.
_SURFACE_FAMILY_PATTERNS = {
    "sky": re.compile(r"\b(?:sky|skies|cloud\w*)\b", re.IGNORECASE),
    "water": re.compile(
        r"\b(?:water|waters|lake\w*|sea|seas|ocean\w*|river\w*|harbou?r\w*|bay|pond\w*|waterfront)\b",
        re.IGNORECASE,
    ),
}


def _neutralize_unlicensed_surface(text, families, keep_sources=()):
    """Strip a surface-family identity the whole-image brief does not place here.

    Returns the neutral literal replacement, or "" when nothing needed changing.
    The tile keeps its own colors as soft continuous tones - the honest prompt
    for pixels too ambiguous to identify. Escape hatch: the user's own request
    words always keep the family.
    """
    keep_text = " ".join(str(source) for source in keep_sources)
    cleaned = str(text)
    hit = False
    for family in families:
        pattern = _SURFACE_FAMILY_PATTERNS.get(str(family))
        if pattern is None or pattern.search(keep_text):
            continue
        if pattern.search(cleaned):
            hit = True
            cleaned = pattern.sub(" ", cleaned)
    if not hit:
        return ""
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;.")
    descriptor = cleaned if cleaned else "soft muted"
    return (
        f"{descriptor} tones in a smooth continuous gradient, kept exactly as "
        "they appear with gentle natural variation"
    )


def _surface_prompt_problem(entry, allow_atmosphere=False):
    """Validate one whole-image surface entry as safe to stamp onto isolated tiles.

    ``allow_atmosphere`` is True for style/time-of-day transforms, where lighting and
    sky words in a surface's target description are the requested result rather than
    an invented flourish.
    """
    if not isinstance(entry, dict):
        return "is not an object"
    identity = str(entry.get("identity", ""))
    for field in ("identity", "target_prompt"):
        text = str(entry.get(field, ""))
        process = _SURFACE_PROCESS_WORD.search(text)
        if process:
            return (
                f'{field} calls the surface a "{process.group(0)}" instead of naming '
                "the real material (for example water, sky, sand, asphalt)"
            )
        obj = _SURFACE_DISCRETE_OBJECT.search(text)
        if obj:
            return (
                f'{field} mentions "{obj.group(0)}" — a surface description must describe '
                "only the surface itself, never nearby objects or their reflections"
            )
    # In a faithful repair/upscale, a non-sky surface may not reference the sky or
    # weather ("reflecting sky"): only the surface itself belongs in its phrase.
    if not allow_atmosphere:
        target = str(entry.get("target_prompt", ""))
        atmosphere = _ATMOSPHERE_TERM.search(target)
        if atmosphere and not _ATMOSPHERE_TERM.search(identity):
            return (
                f'target_prompt mentions "{atmosphere.group(0)}" — a surface description '
                "covers only the surface itself, not the sky or weather"
            )
    return ""


_MAP_PROMPT_FILLER = {"a", "an", "and", "in", "of", "or", "the", "with"}


def _map_prompt_carries_nothing(entry):
    """True when a material's wording is too bare to be worth sharing at all.

    Materials are ADVISORY - a tile still describes everything itself - so a
    phrase that merely names the material with a couple of modifiers ("rough
    white stone wall") is genuinely useful: every tile touching it reuses the
    same words and they stop disagreeing. Only a bare one-word answer ("moss")
    carries nothing. This is a much lower bar than `_map_prompt_is_thin`, which
    guards the AUTHORITATIVE surface path where the phrase replaces a whole tile.
    """
    if not isinstance(entry, dict):
        return True
    words = [
        word
        for word in re.findall(r"[a-z0-9]+", str(entry.get("target_prompt", "")).casefold())
        if word not in _MAP_PROMPT_FILLER
    ]
    return len(words) < 2


def _map_prompt_is_thin(entry):
    """True when a canonical entry's target_prompt only repeats its own name.

    A canonical surface or material exists to give every tile the SAME wording
    for the same thing - "calm, deep teal water with gentle ripples". When the
    whole-image model answers `identity: water, target_prompt: "water"` the
    entry carries nothing: stamped onto a uniform tile it becomes that tile's
    entire prompt. Such an entry is treated as absent, which is exactly how the
    pipeline behaved before canonical maps existed.
    """
    if not isinstance(entry, dict):
        return True
    own = set()
    for field in ("id", "identity"):
        own.update(re.findall(r"[a-z0-9]+", str(entry.get(field, "")).casefold()))
    extra = [
        word
        for word in re.findall(
            r"[a-z0-9]+", str(entry.get("target_prompt", "")).casefold()
        )
        if word not in own and word not in _MAP_PROMPT_FILLER
    ]
    return len(extra) < 2


_SURFACE_IDENTITY_STOPWORDS = {
    "area",
    "region",
    "surface",
    "continuous",
    "the",
    "with",
}


def _canonical_surface_candidates(reference):
    candidates = reference.get("canonical_surfaces")
    if not isinstance(candidates, list):
        return []
    # A thin entry ("water" -> "water") would become a uniform tile's ENTIRE
    # prompt when stamped, which is worse than no canonical surface at all.
    return [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and not _map_prompt_is_thin(candidate)
    ]


# Things a surface can be described as reflecting. The surface contract already
# forbids "reflections of surrounding buildings"; a brief answered "reflections
# of city lights" instead, which the object regex does not cover.
_REFLECTED_SOURCE = re.compile(
    r"\b(?:cit(?:y|ies)|skyline\w*|urban|downtown|street\s?lamp\w*|street\s?light\w*|"
    r"lamp\w*|lantern\w*|neon|headlight\w*|floodlight\w*|signage|window\w*|"
    r"build\w*|tower\w*|skyscraper\w*|boat\w*|ship\w*|bridge\w*|shore\w*|tree\w*)\b",
    re.IGNORECASE,
)


def _strip_reflected_objects(value):
    """Drop "reflecting <something else in the scene>" from a surface phrase.

    A canonical surface is stamped on tiles that contain ONLY that surface, so
    whatever it is said to reflect gets painted there. Every open-water tile of a
    night harbour was told "reflecting city lights" and each one invented its own
    bright highlights - which is precisely what made the tile grid visible on the
    water. A bare "soft reflections" is a real property of the surface and stays.
    """
    text = str(value or "").strip()
    if not text or "reflect" not in text.casefold():
        return text
    clauses = re.split(r"\s*[,;]\s*|\s+\band\b\s+", text)
    kept = [
        clause
        for clause in clauses
        if not (
            re.search(r"reflect", clause, re.IGNORECASE)
            and _REFLECTED_SOURCE.search(clause)
        )
    ]
    if len(kept) == len(clauses):
        return text
    result = ", ".join(clause.strip() for clause in kept if clause.strip())
    return result.strip(" ,;.") or text


def _canonical_surface_text(surface):
    text = _strip_reflected_objects(str(surface.get("target_prompt", "")).strip()).rstrip(" .")
    if text:
        return text
    identity = str(surface.get("identity", "")).strip().rstrip(" .")
    appearance = str(surface.get("source_appearance", "")).strip().rstrip(" .")
    return ", ".join(part for part in (identity, appearance) if part)


# A canonical surface REPLACES a smooth tile's whole description, so it has to
# actually be most of that tile. The nine coarse areas overlap the tile grid, so
# a middle-row tile always clips a little into the top and bottom bands - on a
# lake picture that gave a mountain tile a 20% "water" claim, and the mountain
# description was thrown away for "dark blue water".
_SURFACE_REPLACES_TILE_OVERLAP = 0.5


def _surface_covers_tile(surface):
    """True when this surface may stand in for the whole tile.

    A pixel-measured uniform run is authoritative by construction - it exists to
    cover tiles the model's own location list missed - so it is always allowed.
    A location-derived candidate must measurably dominate the tile. References
    from before overlap scores existed keep their old behaviour.
    """
    if not isinstance(surface, dict):
        return False
    if str(surface.get("selection_source", "")) == "measured_uniform_region":
        return True
    if "spatial_overlap" not in surface:
        return True
    try:
        return float(surface["spatial_overlap"]) >= _SURFACE_REPLACES_TILE_OVERLAP
    except (TypeError, ValueError):
        return True


def _select_canonical_surface(reference, payload, evidence_class):
    candidates = _canonical_surface_candidates(reference)
    if not candidates:
        return None

    requested_id = str(payload.get("surface_id", "")).strip().casefold()
    if requested_id:
        for candidate in candidates:
            if str(candidate.get("id", "")).strip().casefold() == requested_id:
                return candidate

    evidence = " ".join(
        str(payload.get(key, ""))
        for key in ("local_caption", "dominant_region", "surface_prompt")
    ).casefold()
    ranked = []
    for order, candidate in enumerate(candidates):
        identity = str(candidate.get("identity", "")).casefold()
        tokens = {
            token
            for token in re.findall(r"[a-z0-9][a-z0-9'-]*", identity)
            if len(token) >= 4 and token not in _SURFACE_IDENTITY_STOPWORDS
        }
        score = sum(
            bool(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", evidence))
            for token in tokens
        )
        if score:
            ranked.append((score, -order, candidate))
    if ranked:
        ranked.sort(reverse=True)
        if len(ranked) == 1 or ranked[0][0] > ranked[1][0]:
            return ranked[0][2]
    # A measured adjacent uniform-tile run has already proven one continuous
    # surface. It is the strongest deterministic choice available.
    measured = [
        candidate
        for candidate in candidates
        if str(candidate.get("selection_source", "")) == "measured_uniform_region"
    ]
    if len(measured) == 1:
        return measured[0]

    # Otherwise use exact normalized overlap, not model list order. A tie is real
    # ambiguity and must be confirmed by the exact tile rather than silently
    # stamping the first whole-image candidate across it.
    if evidence_class in ("uniform", "sparse") and candidates:
        # References produced before schema v2 did not carry overlap scores.
        # Preserve their deterministic first-candidate behavior; all newly built
        # references below use measured overlap and refuse true ties.
        if not any("spatial_overlap" in candidate for candidate in candidates):
            return candidates[0]
        ranked_overlap = sorted(
            (
                (float(candidate.get("spatial_overlap", 0.0)), candidate)
                for candidate in candidates
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if len(ranked_overlap) == 1:
            return ranked_overlap[0][1]
        if ranked_overlap[0][0] > ranked_overlap[1][0] + 1e-9:
            return ranked_overlap[0][1]
    return None


_OBJECT_IDENTITY_STOPWORDS = _SURFACE_IDENTITY_STOPWORDS | {
    "a",
    "an",
    "and",
    "appearing",
    "area",
    "black",
    "blue",
    "bright",
    "brown",
    "building",
    "clean",
    "concrete",
    "continuous",
    "dark",
    "daylight",
    "detailed",
    "feature",
    "fine",
    "form",
    "glass",
    "grid",
    "gray",
    "green",
    "grey",
    "item",
    "large",
    "light",
    "lit",
    "long",
    "main",
    "metal",
    "metallic",
    "modern",
    "muted",
    "new",
    "object",
    "old",
    "orange",
    "part",
    "pale",
    "pattern",
    "purple",
    "realistic",
    "red",
    "reflection",
    "reflective",
    "region",
    "shadow",
    "sharp",
    "short",
    "small",
    "smooth",
    "soft",
    "source",
    "steel",
    "stone",
    "structure",
    "subject",
    "subtle",
    "surface",
    "teal",
    "that",
    "the",
    "thing",
    "this",
    "uniform",
    "visible",
    "white",
    "with",
    "wood",
    "wooden",
    "yellow",
}


def _normalized_object_token(token):
    """Normalize simple English plurals used by captions and canonical prompts."""
    token = str(token).casefold()
    if len(token) > 5 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 5 and token.endswith(("ches", "shes", "xes", "zes", "ses")):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _object_text_tokens(value):
    """Return concrete object/part words, excluding broad labels and styling."""
    tokens = set()
    for token in re.findall(r"[a-z0-9][a-z0-9'-]*", str(value).casefold()):
        normalized = _normalized_object_token(token)
        if (
            len(normalized) >= 3
            and token not in _OBJECT_IDENTITY_STOPWORDS
            and normalized not in _OBJECT_IDENTITY_STOPWORDS
        ):
            tokens.add(normalized)
    return tokens


def _object_candidate_tokens(candidate):
    """Aliases that can genuinely confirm one canonical object from tile pixels.

    Whole-image models sometimes use a broad identity such as ``building`` while
    placing the useful identity (``wooden cabin``) in ``target_prompt``. Include
    that canonical phrase and the exact per-tile part hint so "cabin", "roof", or
    "balcony" can confirm the same subject. Generic appearance words are excluded
    so a shared word such as "dark" can never confirm an object by itself.
    """
    if not isinstance(candidate, dict):
        return set()
    values = (
        candidate.get("identity", ""),
        candidate.get("target_prompt", ""),
        candidate.get("part_in_this_tile", ""),
    )
    tokens = set()
    for value in values:
        tokens.update(_object_text_tokens(value))
    return tokens


def _object_candidate_label(candidate):
    """Prefer a concrete identity in validation messages over a generic class."""
    identity = str(candidate.get("identity", "")).strip()
    if _object_text_tokens(identity):
        return identity
    target = str(candidate.get("target_prompt", "")).strip()
    return target or identity or str(candidate.get("id", "")).strip()


def _canonical_object_candidates(reference):
    candidates = reference.get("canonical_objects")
    if not isinstance(candidates, list):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def _select_canonical_object(reference, payload):
    """Choose a canonical discrete object ONLY when the tile confirmed it from its pixels.

    Unlike a surface, an object is never auto-selected from location overlap alone: the
    tile must return a matching ``object_id`` or clearly name the object in its local
    evidence. This gives identical cross-tile object wording (Problem C) without ever
    drawing an object into a tile whose pixels do not show it (Problem A stays intact).
    """
    candidates = _canonical_object_candidates(reference)
    if not candidates:
        return None

    requested_id = str(payload.get("object_id", "")).strip().casefold()
    if requested_id:
        for candidate in candidates:
            if str(candidate.get("id", "")).strip().casefold() == requested_id:
                return candidate

    # The tile was asked about both a surface and an object and answered both:
    # surface yes, object no. That is a statement about these pixels, not a
    # missing answer, so token guessing must not overrule it. Open water tiles
    # returning `surface_id: water, object_id: ""` were still handed an entire
    # city skyline, which an edit model then painted into the river.
    if str(payload.get("surface_id", "")).strip():
        return None

    # No explicit confirmation id: accept only a strong identity-token match in the
    # tile's own local evidence, and only when it is unambiguous. A candidate whose
    # part is expected in this tile (part_in_this_tile) confirms on one distinctive
    # token; without that expectation, one generic word ("hair", "gray") is not
    # enough evidence for a whole subject - require two.
    evidence = " ".join(
        str(payload.get(key, ""))
        for key in ("object_prompt", "local_caption", "local_features", "dominant_region")
    )
    if not evidence.strip():
        return None
    evidence_tokens = _object_text_tokens(evidence)
    ranked = []
    for order, candidate in enumerate(candidates):
        tokens = _object_candidate_tokens(candidate)
        if not tokens:
            continue
        score = len(tokens & evidence_tokens)
        try:
            overlap = (
                float(candidate.get("spatial_overlap"))
                if "spatial_overlap" in candidate
                else None
            )
        except (TypeError, ValueError):
            overlap = None
        location_matched = overlap is not None and overlap >= 0.12
        part_expected = bool(str(candidate.get("part_in_this_tile", "")).strip())
        # The brief placed this object elsewhere. Its `target_prompt` still
        # carries ordinary scene words ("water", "mossy", "rocks"), so a tile
        # that merely shares those words would otherwise "confirm" a subject
        # its own pixels never showed - and an edit engine would then paint it.
        # A tile that really does see the object still confirms it by returning
        # `object_id` above; that explicit route is untouched. Legacy references
        # with no measured overlap keep the old token rule.
        if overlap is not None and not location_matched and not part_expected:
            continue
        # A part hint ("glass facade") is real evidence about THIS tile, so one
        # distinctive word still confirms - but only when that word is the part
        # the hint named. A candidate's alias set also contains every word of its
        # long target prompt, and an open water tile confirmed an entire city
        # skyline because both texts happened to contain "lights"; Klein then
        # painted skyscrapers into the river. A coincidence elsewhere in the
        # description is not the expected part turning up.
        part_tokens = _object_text_tokens(candidate.get("part_in_this_tile", ""))
        part_confirmed = bool(part_tokens & evidence_tokens)
        needed = 1 if (part_expected and part_confirmed) else 2
        needed = min(needed, max(1, len(tokens)))
        if score >= needed:
            ranked.append((score, -order, candidate))
    if ranked:
        ranked.sort(reverse=True)
        if len(ranked) == 1 or ranked[0][0] > ranked[1][0]:
            return ranked[0][2]
    return None


def _canonical_object_text(obj):
    """The shared wording to attach to a tile that confirmed this object.

    Prefer the tile's OWN part over the whole-subject description. A brief
    described its subject as "dense cluster of tall buildings ... AND a small
    park with dark green trees", so every tile confirming either half received
    both - and a waterfront park tile, whose part hint correctly read "waterfront
    park with trees and pathways", was handed the tall buildings and Klein built
    them on the lawn.

    This is also what the tile contract already demands of the model itself:
    describe the visible PART of the subject, never the whole subject. Tiles in
    the same area share one part hint, so cross-tile consistency is unaffected.
    """
    part = " ".join(str(obj.get("part_in_this_tile", "")).split()).strip(" ,;.")
    if part:
        return part
    text = str(obj.get("target_prompt", "")).strip().rstrip(" .")
    if text:
        return text
    identity = str(obj.get("identity", "")).strip().rstrip(" .")
    appearance = str(obj.get("source_appearance", "")).strip().rstrip(" .")
    return ", ".join(part for part in (identity, appearance) if part)


class SmartTilePromptResolver:
    CATEGORY = "Smart Upscaler/Prompting"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "prompt_audit", "tile_reference")
    FUNCTION = "resolve"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_prompt": ("STRING", {"forceInput": True, "lazy": True}),
                "tile_reference": ("STRING", {"forceInput": True}),
                "prompt_system": ("SMART_PROMPT_SYSTEM", {"forceInput": True}),
                "negative_fallback": (
                    "STRING",
                    {
                        "default": "changed geometry, duplicated objects, artifacts, seams",
                        "multiline": True,
                    },
                ),
            },
        }

    def check_lazy_status(
        self,
        generated_prompt,
        tile_reference,
        prompt_system,
        negative_fallback,
    ):
        strategy = str(prompt_system.get("prompt_strategy", "task_directed"))
        if strategy != "direct_user" and generated_prompt is None:
            return ["generated_prompt"]
        return []

    def resolve(
        self,
        generated_prompt,
        tile_reference,
        prompt_system,
        negative_fallback,
    ):
        try:
            reference = json.loads(tile_reference)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Tile reference must come from Smart Tile Job Director.") from exc
        if "tile_index" not in reference:
            raise ValueError("Tile reference is missing tile_index.")

        strategy = str(prompt_system.get("prompt_strategy", "task_directed"))
        operation_mode = str(prompt_system.get("operation_mode", "faithful_upscale"))
        user = str(prompt_system.get("user_instruction", "")).strip()
        evidence_class = str(reference.get("evidence_class", "")).strip().lower()
        composition_mode = str(prompt_system.get("composition_mode", "legacy_target_prompt"))

        if composition_mode == "task_directed":
            if generated_prompt is None:
                raise ValueError("Task-directed prompting did not produce an exact-tile target prompt.")
            caption_problem = _source_caption_problem(
                generated_prompt,
                evidence_class,
                True,
                prompt_system.get("known_false_detections", ()),
                reference.get("visual_complexity", ""),
                str(prompt_system.get("task_preset", "")) == "Google Image Enhance",
            )
            if caption_problem:
                raise ValueError(f"Task-directed tile response rejected: {caption_problem}.")
            evidence_payload = _source_caption_payload(generated_prompt)
            raw_response = evidence_payload["raw_response"]
            local_caption = evidence_payload["local_caption"]
            dominant_region = evidence_payload["dominant_region"] or local_caption
            visible_boundaries = evidence_payload["visible_boundaries"]
            corrections_applied = evidence_payload["corrections_applied"]
            local_features = evidence_payload["local_features"]
            target_prompt = evidence_payload["target_prompt"]
            if not target_prompt:
                raise ValueError("Task-directed tile response is missing target_prompt.")
            target_detail = _target_detail(
                target_prompt,
                preserve_viewpoint=evidence_class in ("uniform", "sparse"),
            )
            relocated = _strip_own_tile_location(
                target_detail, reference.get("position", "")
            )
            if relocated != target_detail:
                target_detail = relocated
                corrections_applied = "; ".join(
                    part
                    for part in (
                        str(corrections_applied).strip(" ;"),
                        "tile's own whole-image location removed from the prompt",
                    )
                    if part
                )
            # Canonical continuous surfaces are authoritative ONLY for smooth tiles
            # (uniform/sparse), where the local pixels cannot resolve identity and adjacent
            # tiles must share exact wording so they do not seam. A STRUCTURED tile keeps
            # its own detailed local description (buildings, materials, objects) and must
            # never be overwritten by a surface phrase such as "urban ground surface".
            canonical_surface_id = ""
            canonical_surface_prompt = ""
            if evidence_class in ("uniform", "sparse"):
                canonical_surface = _select_canonical_surface(
                    reference, evidence_payload, evidence_class
                )
                if canonical_surface and not _surface_covers_tile(canonical_surface):
                    # The tile really can see this surface - it named it - but the
                    # surface only clips a corner of the crop. Record it for the
                    # audit and let the tile's own description stand.
                    canonical_surface_id = str(canonical_surface.get("id", "")).strip()
                    corrections_applied = "; ".join(
                        part
                        for part in (
                            str(corrections_applied).strip(" ;"),
                            f"canonical surface \"{canonical_surface_id}\" covers only part of "
                            "this tile; the tile's own description was kept",
                        )
                        if part
                    )
                    canonical_surface = None
                if canonical_surface:
                    canonical_surface_id = str(canonical_surface.get("id", "")).strip()
                    canonical_surface_prompt = _canonical_surface_text(canonical_surface)
            # A canonical discrete object (a bridge/tower/roof spanning tiles) may be reused
            # on any non-uniform tile that confirmed it, for cross-tile object consistency.
            canonical_object_id = ""
            canonical_object_prompt = ""
            if evidence_class != "uniform":
                canonical_object = _select_canonical_object(reference, evidence_payload)
                if canonical_object:
                    canonical_object_id = str(canonical_object.get("id", "")).strip()
                    canonical_object_prompt = _canonical_object_text(canonical_object)
            # For a DENOISE sampler the positive prompt is literal content to
            # render: appending the whole-subject description to a tile that
            # shows only one part of it paints extra copies of the subject
            # (phantom faces in background tiles). Description mode therefore
            # never appends the canonical object text; the tile's own
            # part-specific description carries the consistency. Instruction
            # models keep the append - their edit command frames it as context.
            describe_only = (
                str(prompt_system.get("prompt_format", "instruction_edit"))
                == "description"
            )

            if evidence_class == "uniform":
                if canonical_surface_prompt:
                    target_detail = canonical_surface_prompt
            elif evidence_class == "sparse":
                # A canonical SURFACE is authoritative here, for the seam reason
                # above. A canonical OBJECT is discrete content and may only ADD
                # shared wording: letting it replace the tile's own description
                # flattened seven different waterfront tiles into one identical
                # "modern glass-clad skyscrapers ..." line.
                lead = canonical_surface_prompt or target_detail
                if canonical_object_prompt and not describe_only:
                    lead = _merge_canonical_clause(lead, canonical_object_prompt)
                if lead:
                    target_detail = lead
            else:
                # Structured: the tile's own rich local description leads. Only a confirmed
                # spanning object is appended for continuity; no surface override.
                if canonical_object_prompt and not describe_only:
                    target_detail = _merge_canonical_clause(
                        target_detail, canonical_object_prompt
                    )
            if evidence_class == "uniform" and (
                _AMBIGUOUS_BACKGROUND_ONLY.fullmatch(target_detail)
                or _GENERIC_REGION_PLACEHOLDER.search(target_detail)
            ):
                target_detail = re.sub(
                    r"\bbackground\b",
                    "locally visible uniform surface",
                    target_detail,
                    flags=re.IGNORECASE,
                ).strip()
            if not target_detail:
                raise ValueError("Task-directed target_prompt contained only absence language.")
            view_context = _safe_view_context(reference.get("view_context", "")).rstrip(" .")
            if (
                evidence_class == "uniform"
                and not canonical_surface_prompt
                and view_context
                and not _VIEWPOINT_DESCRIPTION.search(target_detail)
            ):
                target_detail = f"{view_context}. {target_detail}"
            edit_action = str(
                prompt_system.get("edit_action")
                or user
                or prompt_system.get("direct_prompt", "")
                or "Make this image match the described local result."
            ).strip().rstrip(" .")
            # Sky/weather flourishes ("all under a clear sky") survive only when the
            # tile's own caption shows them or the user asked. Canonical surface text
            # was already validated at the whole-image stage, so it is left alone.
            if target_detail and target_detail != canonical_surface_prompt:
                cleared = _strip_unsupported_atmosphere(
                    target_detail,
                    f"{local_caption} {dominant_region}",
                    (edit_action, user),
                )
                if cleared:
                    target_detail = cleared
                elif canonical_surface_prompt:
                    target_detail = canonical_surface_prompt
            # An ambiguous tile's color-based family guess ("dark blue water" on
            # blue mountain mist) loses to the validated brief's placement: when
            # no location-matched candidate licensed the family here, the prompt
            # keeps the literal soft appearance instead of the guessed identity.
            unlicensed_families = reference.get("unlicensed_surface_families") or []
            if (
                unlicensed_families
                and evidence_class in ("uniform", "sparse")
                and not canonical_surface_prompt
            ):
                neutral = _neutralize_unlicensed_surface(
                    target_detail, unlicensed_families, (edit_action, user)
                )
                if neutral:
                    target_detail = neutral
                    corrections_applied = "; ".join(
                        part
                        for part in (
                            str(corrections_applied).strip(" ;"),
                            "unlicensed surface-family claim neutralized",
                        )
                        if part
                    )
            # Reconstruction-artifact words ("debris", "broken", "melted", ...) are repair
            # targets, not real content: strip them from the positive unless the user asked.
            # The prompt must say what SHOULD be there. A repair task also sweeps soft
            # wear words ("minor structural wear").
            aggressive_repair = (
                str(prompt_system.get("task_preset", "")) == "Google Image Enhance"
            )
            repaired_detail = _repair_artifact_language(
                target_detail,
                (edit_action, user, str(prompt_system.get("direct_prompt", ""))),
                aggressive=aggressive_repair,
            )
            if repaired_detail:
                target_detail = repaired_detail
            elif canonical_surface_prompt:
                target_detail = canonical_surface_prompt
            else:
                target_detail = _repair_artifact_language(
                    dominant_region, aggressive=aggressive_repair
                ) or "intact, cleanly rendered local surfaces and structures"
            if evidence_class == "sparse" and visible_boundaries and not describe_only:
                # An instruction about how to treat a cut-off feature. An edit
                # model reads it as a rule; a denoise model reads "edge-cut
                # features ... clipped ... edge" as CONTENT and paints thin
                # stick-like fragments into smooth areas. Description-mode
                # prompts carry only what should be visible.
                target_detail = (
                    f"{target_detail.rstrip(' .')}; edge-cut features remain clipped at the same image edge"
                )
            if describe_only:
                # Denoising samplers (SDXL, Flux, Z-Turbo) render a description;
                # an edit command would just be noise in their text encoding.
                target_detail = _strip_frame_talk(target_detail) or target_detail
                positive = f"{target_detail.rstrip(' .')}."
            else:
                positive = f"{edit_action}. {target_detail}."
            suffix = str(prompt_system.get("prompt_suffix", "")).strip().strip(",.")
            if suffix:
                positive = f"{positive.rstrip(' .')}, {suffix}."
            if str(prompt_system.get("color_words", "keep")) == "strip":
                stripped_positive = _strip_color_words(
                    positive,
                    (edit_action, user, str(prompt_system.get("direct_prompt", ""))),
                )
                positive = stripped_positive or (
                    f"{edit_action}. Preserve the source-visible local content and geometry."
                )

            if evidence_class == "uniform":
                local_evidence = dominant_region
                evidence_guard = "task_directed_uniform"
            elif evidence_class == "sparse":
                local_evidence = dominant_region
                if visible_boundaries:
                    local_evidence = f"{local_evidence.rstrip(' .')}; {visible_boundaries}"
                evidence_guard = "task_directed_sparse"
            else:
                local_evidence = local_caption
                evidence_guard = "task_directed_exact"

            negative_parts = [
                "new objects",
                "new regions",
                "new boundaries",
                "changed viewpoint",
                "changed layout",
                "duplicated objects",
                "unrequested material substitution",
                "inconsistent cross-tile appearance",
            ]
            for part in re.split(r"[,;]", str(negative_fallback)):
                part = part.strip()
                if part and not any(
                    part.lower() == existing.lower() for existing in negative_parts
                ):
                    negative_parts.append(part)
            negative = ", ".join(negative_parts)
            response = json.dumps(
                {
                    "strategy": strategy,
                    "operation_mode": operation_mode,
                    "composition_mode": composition_mode,
                    "evidence_guard": evidence_guard,
                    "raw_source_response": raw_response,
                    "local_source_evidence": local_evidence,
                    "corrections_applied": corrections_applied,
                    "edit_action": edit_action,
                    "canonical_surface_id": canonical_surface_id,
                    "canonical_surface_prompt": canonical_surface_prompt,
                    "canonical_object_id": canonical_object_id,
                    "canonical_object_prompt": canonical_object_prompt,
                    "local_features": local_features,
                    "canonical_surface_override": bool(
                        canonical_surface_prompt
                        and evidence_class in ("uniform", "sparse")
                    ),
                    "canonical_object_applied": bool(canonical_object_prompt),
                    "target_detail": target_detail,
                    "positive_prompt": positive,
                    "negative_prompt": negative,
                }
            )

        elif strategy == "direct_user":
            positive = user or str(prompt_system.get("direct_prompt", "")).strip()
            if not positive:
                raise ValueError("No-analysis prompting requires a user request or preset action.")
            suffix = str(prompt_system.get("prompt_suffix", "")).strip().strip(",.")
            if suffix:
                positive = f"{positive.rstrip(' .')}, {suffix}."
            if str(prompt_system.get("color_words", "keep")) == "strip":
                positive = _strip_color_words(
                    positive,
                    (user, str(prompt_system.get("direct_prompt", ""))),
                ) or positive
            negative = str(negative_fallback).strip()
            response = json.dumps(
                {
                    "strategy": "direct_user",
                    "operation_mode": operation_mode,
                    "positive_prompt": positive,
                    "negative_prompt": negative,
                }
            )
        else:
            raise ValueError(
                f"Unsupported prompt strategy '{strategy}'. The Prompt Director produces "
                "task-directed analysis or the direct user-request bypass."
            )

        if not positive:
            raise ValueError(f"Tile {reference['tile_index']} did not produce a usable positive prompt.")
        return positive, negative, response, tile_reference
