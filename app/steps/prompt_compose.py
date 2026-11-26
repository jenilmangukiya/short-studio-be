"""Prompt composition utilities for Shorts Studio.

Combines scene-specific image prompts, recurring subject anchors, visual styles,
environment settings, camera directions, focal points, and fixed vertical framing rules
into a unified final prompt for image generation.
"""

from __future__ import annotations

import re
from typing import Any

from app.styles.adapter import adapt_character, adapt_concept, adapt_environment, adapt_object
from app.styles.registry import (
    VisualStyleResolutionError,
    get_style,
    is_valid_style_id,
    resolve_visual_style_preset,
)
from app.styles.schemas import VisualStylePreset

# Fixed composition and framing rules appended to every final prompt
COMPOSITION_RULES: str = (
    "Vertical 9:16, main subject in the upper-middle of the frame, "
    "calm uncluttered lower third, no text, no letters, no logos, no watermark."
)

# Maximum sensible length for a code-assembled final prompt
MAX_FINAL_PROMPT_LENGTH: int = 1500

# Regex patterns for detecting scarcity, famine, and crop failure
SCARCITY_CROP_PATTERNS: tuple[str, ...] = (
    r"\b(?:harvests?|crops?)\s+(?:failed|failing|ruined|damaged|destroyed|rotted|spoiled|withered|curled)\b",
    r"\b(?:failed|failing|ruined|damaged|spoiled|meager|scanty|sparse)\s+(?:harvests?|crops?)\b",
    r"\bcrops?\s+(?:suffered|froze|dying)\b",
    r"\b(?:crop|harvest)\s+failure\b",
    r"\blittle\s+usable\s+harvest\b",
    r"\bspoilage\b",
)

SCARCITY_FOOD_PATTERNS: tuple[str, ...] = (
    r"\bfood\s+(?:grew\s+)?(?:scarce|scarcity|shortage|ran\s+out)\b",
    r"\b(?:famine|starvation|starving|hunger|hungry|malnutrition)\b",
    r"\b(?:empty|bare|depleted)\s+(?:\w+\s+)*(?:baskets?|shelves|storerooms?|granar(?:y|ies)|pantr(?:y|ies))\b",
    r"\bdwindling\s+(?:\w+\s+)*(?:provisions|food|supplies|rations)\b",
    r"\bsparse\s+food\b",
)

CONCEPT_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "into", "onto", "over", "under",
    "about", "shows", "showing", "look", "looking", "clean", "upper", "lower", "frame",
    "middle", "background", "foreground", "through", "across", "along", "against", "around",
    "behind", "between", "while", "where", "small", "large", "their", "other", "another",
    "visual", "moment", "view", "scene", "beat", "shot", "angle", "high", "deep", "dramatic",
    "subtle", "distinct", "style", "render", "diagram", "cutaway", "schematic", "illustration",
    "illustrating", "simplified", "narrative", "clear", "callouts", "some", "than", "more",
    "less", "rather", "link", "links", "below", "above", "beside", "within", "become", "becomes",
})


def extract_scarcity_negative_constraints(text: str) -> list[str]:
    """Extract targeted negative anchors preventing contradictory visuals of abundance.

    Applied strictly to beats describing crop failures, famine, or food scarcity.
    """
    negatives: list[str] = []
    if not text:
        return negatives

    is_crop_failure = any(re.search(pat, text, re.IGNORECASE) for pat in SCARCITY_CROP_PATTERNS)
    is_food_scarcity = any(re.search(pat, text, re.IGNORECASE) for pat in SCARCITY_FOOD_PATTERNS)

    if is_crop_failure:
        for term in (
            "no abundant harvest",
            "no lush crops",
            "no healthy vegetables",
            "no overflowing produce",
            "no vibrant green fields",
        ):
            if term not in negatives:
                negatives.append(term)

    if is_food_scarcity:
        for term in (
            "no abundant food",
            "no full baskets",
            "no overflowing pantry",
            "no plentiful feast",
        ):
            if term not in negatives:
                negatives.append(term)

    return negatives


def slice_concept_for_beat(
    concept: Any,
    beat_context: Any,
) -> tuple[list[str], str | None, str | None]:
    """Slice a multi-stage concept to identify elements and text relevant to a specific beat.

    Returns:
        tuple of (sliced_graphical_elements, focused_explanation, focused_analogy)
    """
    def _extract_val(obj: Any, field: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)

    raw_elements = _extract_val(concept, "graphical_elements") or []
    explained = str(_extract_val(concept, "concept_explained") or "").strip()
    analogy = str(_extract_val(concept, "visual_analogy") or "").strip()

    # If the concept is already atomic/compact (<= 2 elements), preserve all elements
    if len(raw_elements) <= 2:
        return list(raw_elements), None, None

    # Collect beat contextual intent
    desc = str(_extract_val(beat_context, "visual_description", "") or "").strip()
    focal = str(_extract_val(beat_context, "focal_point", "") or "").strip()
    camera = _extract_val(beat_context, "camera")
    if not focal and isinstance(camera, dict):
        focal = str(camera.get("focal_point_description", "") or "").strip()
    elif not focal and hasattr(camera, "focal_point_description"):
        focal = str(getattr(camera, "focal_point_description", "") or "").strip()
    purpose = str(_extract_val(beat_context, "visual_purpose", "") or "").strip()

    beat_narr = str(
        _extract_val(beat_context, "narration_text", "")
        or _extract_val(beat_context, "narration_slice", "")
        or _extract_val(beat_context, "beat_narration", "")
        or ""
    ).strip()
    scene_narr = str(
        _extract_val(beat_context, "scene_narration", "")
        or _extract_val(beat_context, "narration", "")
        or ""
    ).strip()

    if not beat_narr:
        cand_narr = str(_extract_val(beat_context, "narration", "") or "").strip()
        explicit_scene_narr = str(_extract_val(beat_context, "scene_narration", "") or "").strip()
        if cand_narr and cand_narr != explicit_scene_narr:
            beat_narr = cand_narr

    is_beat_narr_empty_or_scene_dup = (
        not beat_narr
        or (scene_narr and beat_narr.strip() == scene_narr.strip() and len(scene_narr.split()) > 8)
    )

    primary_intent_text = f"{desc} {focal} {purpose}".lower()
    if not is_beat_narr_empty_or_scene_dup and beat_narr:
        primary_intent_text += f" {beat_narr.lower()}"

    primary_intent_words = set(re.findall(r"\b[a-z]{3,}\b", primary_intent_text)) - CONCEPT_STOPWORDS
    scene_narr_words = set(re.findall(r"\b[a-z]{3,}\b", scene_narr.lower())) - CONCEPT_STOPWORDS

    scored: list[tuple[int, str]] = []
    for elem in raw_elements:
        elem_words = set(re.findall(r"\b[a-z]{3,}\b", str(elem).lower())) - CONCEPT_STOPWORDS
        score = 0
        for ew in elem_words:
            if ew in focal.lower() or ew in desc.lower():
                score += 5
            elif ew in purpose.lower() or (not is_beat_narr_empty_or_scene_dup and ew in beat_narr.lower()):
                score += 3
            elif any(ew.startswith(bw[:4]) or bw.startswith(ew[:4]) for bw in primary_intent_words if len(bw) >= 4 and len(ew) >= 4):
                score += 2
            elif ew in scene_narr_words:
                score += 1 if is_beat_narr_empty_or_scene_dup else 2
        scored.append((score, str(elem)))

    scored.sort(key=lambda x: x[0], reverse=True)
    matching = [elem for score, elem in scored if score > 0]

    if matching:
        top_score = scored[0][0]
        sliced_elements = [scored[0][1]]
        if len(scored) > 1 and scored[1][0] > 0 and scored[1][0] >= top_score * 0.45:
            sliced_elements.append(scored[1][1])
    else:
        sliced_elements = []

    # Identify downstream elements strictly after the selected elements in the causal chain
    selected_indices = [
        idx for idx, elem in enumerate(raw_elements) if elem in sliced_elements
    ]
    max_selected_idx = max(selected_indices) if selected_indices else -1
    downstream_elements = [
        raw_elements[idx]
        for idx in range(len(raw_elements))
        if idx > max_selected_idx
    ]
    unrelated_downstream_words: set[str] = set()
    for de in downstream_elements:
        unrelated_downstream_words.update(
            set(re.findall(r"\b[a-z]{3,}\b", str(de).lower())) - CONCEPT_STOPWORDS
        )
    selected_words: set[str] = set()
    for se in sliced_elements:
        selected_words.update(
            set(re.findall(r"\b[a-z]{3,}\b", str(se).lower())) - CONCEPT_STOPWORDS
        )
    unrelated_downstream_words = unrelated_downstream_words - selected_words - primary_intent_words

    # Focused explanation: select matching sentence prioritizing primary visual intent
    focused_explained: str | None = None
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", explained) if s.strip()]
    if len(sentences) > 1:
        s_scored: list[tuple[int, str]] = []
        for s in sentences:
            s_words = set(re.findall(r"\b[a-z]{3,}\b", s.lower())) - CONCEPT_STOPWORDS
            score = sum(4 for w in s_words if w in focal.lower() or w in desc.lower())
            score += sum(2 for w in s_words if w in purpose.lower() or (not is_beat_narr_empty_or_scene_dup and w in beat_narr.lower()))
            if is_beat_narr_empty_or_scene_dup:
                score += sum(1 for w in s_words if w in scene_narr_words and w not in unrelated_downstream_words)
            else:
                score += sum(1 for w in s_words if w in scene_narr_words)
            s_scored.append((score, s))
        s_scored.sort(key=lambda x: x[0], reverse=True)
        if s_scored[0][0] > 0:
            focused_explained = s_scored[0][1]

    # Focused analogy: guard against compound analogy leakage
    focused_analogy: str | None = None
    a_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", analogy) if s.strip()]
    if a_sentences:
        clean_units: list[tuple[int, int, str]] = []
        for s in a_sentences:
            s_words = set(re.findall(r"\b[a-z]{3,}\b", s.lower())) - CONCEPT_STOPWORDS
            pos = sum(4 for w in s_words if w in focal.lower() or w in desc.lower())
            pos += sum(3 for w in s_words if w in selected_words)
            pos += sum(1 for w in s_words if w in primary_intent_words)
            neg = sum(5 for w in s_words if w in unrelated_downstream_words)

            if pos > 0 and neg == 0:
                clean_units.append((pos, len(s), s))
                continue

            parts = re.split(
                r"(,\s*(?:then\s+(?:shows|illustrates)?|and then|while|before)\s+|\s+before\s+)",
                s,
                flags=re.IGNORECASE,
            )
            if len(parts) > 1:
                for end_idx in range(0, len(parts), 2):
                    prefix = "".join(parts[: end_idx + 1]).strip().rstrip(".")
                    if len(prefix.split()) < 4:
                        continue
                    p_words = set(re.findall(r"\b[a-z]{3,}\b", prefix.lower())) - CONCEPT_STOPWORDS
                    p_pos = sum(4 for w in p_words if w in focal.lower() or w in desc.lower())
                    p_pos += sum(3 for w in p_words if w in selected_words)
                    p_pos += sum(1 for w in p_words if w in primary_intent_words)
                    p_neg = sum(5 for w in p_words if w in unrelated_downstream_words)

                    if p_pos > 0 and p_neg == 0:
                        clean_units.append((p_pos, len(prefix), prefix))

        if clean_units:
            clean_units.sort(key=lambda x: (x[0], x[1]), reverse=True)
            best = clean_units[0][2].strip().rstrip(".")
            if best:
                focused_analogy = f"{best[0].upper()}{best[1:]}."
        else:
            focused_analogy = None

    return sliced_elements, focused_explained, focused_analogy


def _has_bedroom_context(
    entities: dict[str, Any],
    creative_direction: dict[str, Any] | None = None,
    video_topic: str | None = None,
    art_dir: dict[str, Any] | None = None,
) -> bool:
    """Check whether the video topic or production entities actually involve a bedroom or sleep setting."""
    def _extract_val(obj: Any, field: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)

    # 1. Check environments
    envs = entities.get("environments") or []
    for e in envs:
        e_str = (
            f"{_extract_val(e, 'id', '')} {_extract_val(e, 'name', '')} {_extract_val(e, 'spatial_layout', '')}"
        ).lower()
        if any(term in e_str for term in ("bedroom", "bedside", "bedding", "mattress", "pillow", "duvet")):
            return True

    # 2. Check concepts
    concepts = entities.get("concepts") or entities.get("concepts_and_diagrams") or []
    for cp in concepts:
        cp_str = (
            f"{_extract_val(cp, 'name', '')} {_extract_val(cp, 'concept_explained', '')} {_extract_val(cp, 'visual_analogy', '')}"
        ).lower()
        if any(term in cp_str for term in ("bedroom", "bedside", "bedding", "sleeper", "sleep onset", "in the bed", "into the bed")):
            return True

    # 3. Check art direction
    if art_dir:
        ad_str = (
            f"{art_dir.get('rendering_texture', '')} {' '.join(str(a) for a in art_dir.get('negative_anchors', []))}"
        ).lower()
        if any(term in ad_str for term in ("bedroom", "bedding", "sleeper")):
            return True

    # 4. Check creative direction and topic
    cd_dict = (
        creative_direction.model_dump()
        if hasattr(creative_direction, "model_dump")
        else dict(creative_direction) if creative_direction else {}
    )
    story_angle = str(cd_dict.get("storytelling_angle", "")).lower()
    thesis = str(cd_dict.get("central_thesis", "")).lower()
    topic_str = str(video_topic or "").lower()

    return any(
        term in story_angle or term in thesis or term in topic_str
        for term in ("bedroom", "sleep", "hypnic", "sleeper", "insomnia", "bedtime")
    )


INTERIOR_LOCATION_REGEX = re.compile(
    r"\b(?:"
    r"interior|inside|indoor|"
    r"storeroom|cellar|room|bedroom|cabin|cockpit|"
    r"hall|kitchen|pantry|granary|office|laboratory|lab|"
    r"parlor|attic|basement|workshop|hangar|shed|inn|tavern|"
    r"cottage\s+interior|house\s+interior"
    r")\b",
    re.IGNORECASE,
)

OUTDOOR_LANDSCAPE_REGEX = re.compile(
    r"\b(?:"
    r"landscape|fields?|crops?|furrows?|slopes?|mountain|volcano|"
    r"valley|hills?|coastline|shore|ocean|sea|plains?|pasture|"
    r"ridges?|open\s+sky|aerial|exterior"
    r")\b",
    re.IGNORECASE,
)

def slice_environment_for_beat(
    env: Any,
    beat_context: Any,
) -> tuple[list[str], bool, str | None, list[str]]:
    """Slice environment properties to select only elements appropriate for a specific beat.

    Prevents outdoor landscape layouts and outdoor lighting from contaminating interior shots,
    prevents wide-angle framing from overriding close-ups, and refines persistent props to those
    relevant to the beat composition.

    Returns:
        tuple of (relevant_props, include_spatial_layout, local_setting, additional_negatives)
    """
    def _extract_val(obj: Any, field: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)

    intent = str(_extract_val(beat_context, "visual_intent", "") or "").strip().lower()
    shot_type = str(_extract_val(beat_context, "shot_type", "") or "").strip().lower()
    camera = _extract_val(beat_context, "camera")
    if isinstance(camera, dict) and camera.get("shot_type"):
        shot_type = str(camera["shot_type"]).lower()
    elif hasattr(camera, "shot_type") and getattr(camera, "shot_type", None):
        shot_type = str(getattr(camera, "shot_type")).lower()

    desc = str(_extract_val(beat_context, "visual_description", "") or "").strip()
    focal = str(_extract_val(beat_context, "focal_point", "") or "").strip()
    local_env = str(_extract_val(beat_context, "environment", "") or "").strip()

    combined_context = f"{desc} {focal} {local_env}".lower()

    raw_props = _extract_val(env, "persistent_props") or []
    spatial_layout = str(_extract_val(env, "spatial_layout", "") or "").strip()
    env_name = str(_extract_val(env, "name", "") or "").strip()
    is_parent_outdoor = bool(OUTDOOR_LANDSCAPE_REGEX.search(f"{env_name} {spatial_layout}"))

    relevant_props = [str(p) for p in raw_props]
    include_spatial_layout = True
    local_setting: str | None = None
    additional_negatives: list[str] = []

    # 1. Interior Isolation Check
    is_interior_shot = (
        shot_type in ("interior",)
        or bool(INTERIOR_LOCATION_REGEX.search(combined_context))
    )

    if is_interior_shot:
        include_spatial_layout = False
        if local_env:
            local_setting = local_env
        elif is_parent_outdoor:
            if INTERIOR_LOCATION_REGEX.search(desc):
                m_interior = INTERIOR_LOCATION_REGEX.search(desc)
                local_setting = m_interior.group(0).capitalize() if m_interior else "Interior setting"
            else:
                local_setting = "Interior setting"

        # Filter out persistent props describing outdoor landscape features
        relevant_props = [
            p for p in relevant_props
            if not OUTDOOR_LANDSCAPE_REGEX.search(p)
            and any(w in combined_context for w in re.findall(r"\b[a-z]{3,}\b", p.lower()))
        ]
        if is_parent_outdoor:
            additional_negatives.append("no outdoor landscape")
            additional_negatives.append("no outdoor sky")

    # 2. Close-up / Macro / Detail Focus Isolation
    is_close_up = (
        shot_type in ("close-up", "extreme-close-up", "macro", "close_up", "extreme_close_up")
        or intent == "detail_focus"
    )
    if is_close_up:
        include_spatial_layout = False
        # Only keep persistent props that are explicitly mentioned or relevant to focal/desc
        relevant_props = [
            p for p in relevant_props
            if any(w in f"{focal} {desc}".lower() for w in re.findall(r"\b[a-z]{3,}\b", p.lower()))
        ]

    # 3. General prop relevance slicing (prevent overloading non-wide beats with irrelevant props)
    is_wide_establishing = (
        shot_type in ("wide", "exterior", "aerial")
        and intent in ("establishing", "environmental_context", "action_event")
    )
    if not is_wide_establishing and len(relevant_props) > 2:
        context_words = set(re.findall(r"\b[a-z]{3,}\b", combined_context))
        filtered = [
            p for p in relevant_props
            if any(w in context_words for w in re.findall(r"\b[a-z]{3,}\b", p.lower()))
        ]
        if filtered:
            relevant_props = filtered
        else:
            relevant_props = relevant_props[:1]

    return relevant_props, include_spatial_layout, local_setting, additional_negatives


def _clean_segment(text: str) -> str:
    """Normalize whitespace and ensure clean sentence-ending punctuation."""
    t = text.strip()
    if not t:
        return ""
    # Strip any trailing commas, semicolons, colons, or whitespace
    t = re.sub(r"[,;:\s]+$", "", t)
    # Deduplicate multiple consecutive trailing periods (e.g. '..' -> '.')
    t = re.sub(r"\.+$", ".", t)
    # If not ending in terminal punctuation (. ! ?), add a period
    if not re.search(r"[.!?]$", t):
        t += "."
    return t


def _clean_rule_text(rule_text: str) -> str:
    """Clean illustrative example clauses (e.g. '(e.g. ...)') from style rules while preserving other parentheticals."""
    if not rule_text:
        return ""
    cleaned = re.sub(r"\s*\((?:e\.g\.|for example)[^)]*\)", "", rule_text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_world_negative_constraints(
    art_dir: dict[str, Any],
    creative_direction: dict[str, Any] | None = None,
    is_standalone_concept: bool = False,
) -> list[str]:
    """Extract story-wide world, era, and visual negative constraints.

    These constraints apply across all scenes in the video (including scenes
    without a bound character), preventing anachronisms (e.g. modern clothing
    in a historical era) and visual contradictions.
    """
    world_negatives: list[str] = []

    # 1. Art direction negative anchors
    raw_anchors = art_dir.get("negative_anchors") or []
    for item in raw_anchors:
        s = str(item).strip()
        if not s:
            continue
        lower = s.lower()
        if is_standalone_concept and any(term in lower for term in ("sleeper", "character", "bedroom", "person")):
            continue
        if s not in world_negatives:
            world_negatives.append(s)

    # 2. Art direction negative prompt tokens (comma-separated)
    raw_tokens = art_dir.get("negative_prompt_tokens") or ""
    if isinstance(raw_tokens, str) and raw_tokens.strip():
        for tok in raw_tokens.split(","):
            t = tok.strip()
            if not t:
                continue
            lower_t = t.lower()
            if is_standalone_concept and any(term in lower_t for term in ("sleeper", "bedroom")):
                continue
            if t not in world_negatives:
                world_negatives.append(t)

    # 3. Creative direction world/era visual constraints
    if creative_direction:
        cd_dict = (
            creative_direction.model_dump()
            if hasattr(creative_direction, "model_dump")
            else dict(creative_direction) if isinstance(creative_direction, dict) else {}
        )
        constraints = cd_dict.get("creative_constraints") or []
        for c in constraints:
            c_str = str(c).strip()
            if not c_str:
                continue
            lower_c = c_str.lower()
            is_negative = lower_c.startswith(("never ", "no ", "do not ", "don't ", "avoid ", "without "))
            is_world_visual = any(w in lower_c for w in (
                "depict", "show", "clothing", "modern", "technology", "era", "vehicle",
                "accessories", "setting", "architecture", "photorealism", "vanish", "disappear"
            ))
            if is_negative and is_world_visual:
                if "fictional" in lower_c or "representative" in lower_c:
                    continue
                clean_c = c_str.rstrip(".")
                if clean_c not in world_negatives:
                    world_negatives.append(clean_c)

    return world_negatives


def adapt_visual_style(
    visual_style: str | None,
    environment: str = "",
    shot_type: str = "",
    visual_technique: str = "",
) -> str:
    """Adapt the overall visual style to the specific scene context.

    Prevents generic studio pastel backdrops or shallow blur from overriding
    specific environments (e.g. data centers, space, aircraft cabins, landscapes)
    or technical diagram / wide framing shots.
    """
    if not visual_style:
        return ""

    style_clean = visual_style.strip()
    if not style_clean:
        return ""

    has_specific_env = bool(environment.strip()) or any(
        w in environment.lower()
        for w in (
            "server",
            "data center",
            "cloud",
            "sky",
            "space",
            "cabin",
            "interior",
            "exterior",
            "landscape",
            "ocean",
            "room",
            "hall",
            "facility",
            "factory",
            "chassis",
            "rack",
        )
    )
    is_wide_or_technical = shot_type.lower() in (
        "wide",
        "aerial",
        "exterior",
        "cutaway",
        "cross-section",
        "diagram",
        "top-down",
        "in-context",
    ) or any(
        t in visual_technique.lower()
        for t in ("diagram", "cutaway", "schematic", "technical", "cross-section", "macro-photo")
    )

    # Split by comma to evaluate individual style descriptors
    parts = [p.strip() for p in style_clean.split(",") if p.strip()]
    kept_parts: list[str] = []

    for part in parts:
        low = part.lower()
        # Remove generic pastel/gradient backdrop when a real environment or technical shot is present
        if (has_specific_env or is_wide_or_technical) and (
            "pastel gradient" in low or "gradient background" in low or "simple pastel" in low
        ):
            continue
        # Remove shallow depth of field on wide vistas, architectural halls, or exploded diagrams
        if is_wide_or_technical and "shallow depth of field" in low:
            continue
        kept_parts.append(part)

    if not kept_parts:
        return style_clean

    return ", ".join(kept_parts)


def _clean_conceptual_analogy(text: str) -> str:
    """Sanitize concept visual analogy for standalone conceptual beats.

    Ensures only a single figure or silhouette is referenced, and converts physical
    bedding/furniture references into abstract visual equivalents.
    """
    if not text:
        return ""
    s = text.strip()

    # Remove leading physical bedroom actions (e.g. "The sleeper gently settles into the bed while...")
    s = re.sub(
        r"^[Tt]he\s+sleeper\s+gently\s+settles\s+into\s+the\s+bed\s+while\s+",
        "",
        s,
    )
    if s and s[0].islower():
        s = s[0].upper() + s[1:]

    # Prevent duplicate figures: transform "passes through the duvet as the sleeper gives..."
    s = re.sub(
        r"\bthrough\s+the\s+duvet\s+as\s+the\s+sleeper\s+gives\b",
        "through a single simplified body silhouette during",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\bthrough\s+the\s+duvet\b",
        "through a single simplified body silhouette",
        s,
        flags=re.IGNORECASE,
    )

    # Transform bedding and furniture to abstract equivalents
    s = re.sub(r"\bback\s+into\s+the\s+pillow\b", "back to rest", s, flags=re.IGNORECASE)
    s = re.sub(r"\binto\s+the\s+pillow\b", "to rest", s, flags=re.IGNORECASE)
    s = re.sub(r"\binto\s+the\s+bed\b", "into rest", s, flags=re.IGNORECASE)
    s = re.sub(r"\bon\s+the\s+mattress\b", "in abstract space", s, flags=re.IGNORECASE)
    s = re.sub(r"\bon\s+the\s+bed\b", "in abstract space", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(duvet|pillow|mattress|bedding)\b", "", s, flags=re.IGNORECASE)

    # Standardize character references to a single body silhouette
    s = re.sub(r"\ba\s+simple\s+body\s+silhouette\b", "a single body silhouette", s, flags=re.IGNORECASE)
    s = re.sub(r"\bthe\s+sleeper\b", "the silhouette", s, flags=re.IGNORECASE)
    s = re.sub(r"\bthe\s+sleeper's\b", "the silhouette's", s, flags=re.IGNORECASE)

    # Clean whitespace and punctuation
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;])", r"\1", s)
    s = re.sub(r",\s*,", ",", s)
    return s.strip()


def _clean_conceptual_styling(text: str) -> str:
    """Sanitize styling/explanation text so it complements the visual diagram without repeating the visual silhouette or character."""
    if not text:
        return ""
    s = text.strip()

    # In styling/explanation, replace "as the sleeper drifts off..." with bodily/sensory transition
    s = re.sub(
        r"\bAs\s+the\s+sleeper\s+drifts\s+off\s+and\s+muscles\s+relax\b",
        "As muscles relax during the transition to sleep",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\bthe\s+sleeper\s+drifts\s+off\b",
        "the body drifts into rest",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\bas\s+a\s+person\s+transitions\s+into\s+sleep\b",
        "during the sleep-onset transition",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\bthe\s+sleeper\b", "the resting body", s, flags=re.IGNORECASE)
    s = re.sub(r"\bthe\s+sleeper's\b", "the body's", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(duvet|pillow|mattress|bedding|bed)\b", "", s, flags=re.IGNORECASE)

    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;])", r"\1", s)
    s = re.sub(r",\s*,", ",", s)
    return s.strip()


def _resolve_preset(
    preset: VisualStylePreset | dict[str, Any] | str | None = None,
    context: Any = None,
) -> VisualStylePreset | None:
    """Resolve a VisualStylePreset from an explicit argument or a context dict/object with fail-closed rules."""
    if preset is not None:
        if isinstance(preset, VisualStylePreset):
            return preset
        if isinstance(preset, dict):
            return resolve_visual_style_preset(snapshot=preset)
        if isinstance(preset, str):
            target_version = None
            if isinstance(context, dict):
                target_version = context.get("visual_style_version")
            elif context is not None:
                target_version = getattr(context, "visual_style_version", None)
            return resolve_visual_style_preset(style_id=preset, style_version=target_version)

    # Extract from context
    snap = None
    style_id = None
    style_version = None

    if isinstance(context, dict):
        snap = context.get("visual_style_snapshot") or context.get("_visual_style_snapshot")
        style_id = context.get("visual_style_id") or context.get("_visual_style_id")
        style_version = context.get("visual_style_version") or context.get("_visual_style_version")
    elif context is not None:
        snap = getattr(context, "visual_style_snapshot", None)
        style_id = getattr(context, "visual_style_id", None)
        style_version = getattr(context, "visual_style_version", None)

    return resolve_visual_style_preset(
        snapshot=snap,
        style_id=style_id,
        style_version=style_version,
    )


def _compose_connected_final_prompt(
    scene: Any,
    video_bible: dict[str, Any] | Any,
    creative_direction: dict[str, Any] | Any | None = None,
    preset: VisualStylePreset | dict[str, Any] | str | None = None,
    video_topic: str | None = None,
) -> str:
    """Compose an image prompt deterministically from Video Bible entities and directorial intent.

    Applies:
    - Global Art Direction (visual genre, lighting philosophy, rendering texture) or visual style preset.
    - ONLY the entities explicitly bound to this specific beat (character, environment, props, concept),
      adapted to the visual style preset when active, with multi-stage concepts sliced to beat-relevant elements.
    - Camera direction, framing, and primary visual focal point.
    - Narrative transition rationale (e.g. cutaway framing).
    - Art direction negative anchors, preset negative anchors, scarcity constraints, and vertical 9:16 safe-zone rules.
    """
    resolved_preset = _resolve_preset(preset, context=scene)
    def _extract(obj: Any, field: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)

    # 1. Normalize Video Bible data
    vb_dict = (
        video_bible.model_dump()
        if hasattr(video_bible, "model_dump")
        else dict(video_bible)
    )
    entities = vb_dict.get("entities") or {}
    if hasattr(entities, "model_dump"):
        entities = entities.model_dump()
    art_dir = vb_dict.get("art_direction") or {}
    if hasattr(art_dir, "model_dump"):
        art_dir = art_dir.model_dump()

    chars_by_id = {
        str(_extract(c, "id")): c for c in (entities.get("characters") or []) if _extract(c, "id")
    }
    envs_by_id = {
        str(_extract(e, "id")): e for e in (entities.get("environments") or []) if _extract(e, "id")
    }
    raw_objs = entities.get("objects") or entities.get("objects_and_props") or []
    objs_by_id = {
        str(_extract(o, "id")): o for o in raw_objs if _extract(o, "id")
    }
    raw_concepts = entities.get("concepts") or entities.get("concepts_and_diagrams") or []
    concepts_by_id = {
        str(_extract(cp, "id")): cp for cp in raw_concepts if _extract(cp, "id")
    }

    # 2. Extract scene/beat bindings
    raw_bindings = _extract(scene, "entity_bindings")
    if hasattr(raw_bindings, "model_dump"):
        bindings = raw_bindings.model_dump()
    elif hasattr(raw_bindings, "dict"):
        bindings = raw_bindings.dict()
    elif isinstance(raw_bindings, dict):
        bindings = raw_bindings
    else:
        bindings = {}

    char_id = str(bindings.get("character_id") or "").strip() or None
    env_id = str(bindings.get("environment_id") or "").strip() or None
    concept_id = str(bindings.get("concept_id") or "").strip() or None
    raw_obj_ids = bindings.get("object_ids") or []
    obj_ids = [str(o).strip() for o in raw_obj_ids if str(o).strip()]

    # Enforce strict character prohibition when human_presence is 'none' or no characters in catalog
    cd_dict = (
        creative_direction.model_dump()
        if hasattr(creative_direction, "model_dump")
        else dict(creative_direction) if creative_direction else {}
    )
    protagonist_cfg = cd_dict.get("protagonist_config") or {}
    if hasattr(protagonist_cfg, "model_dump"):
        protagonist_cfg = protagonist_cfg.model_dump()
    expected_presence = str(protagonist_cfg.get("human_presence", "")).lower()
    if expected_presence == "none" or not chars_by_id:
        char_id = None

    # 3. Extract camera direction
    camera_data = _extract(scene, "camera")
    shot_type = ""
    camera_angle = ""
    focal_point = ""
    if isinstance(camera_data, dict):
        shot_type = str(camera_data.get("shot_type") or "").strip()
        camera_angle = str(camera_data.get("camera_angle") or "").strip()
        focal_point = str(camera_data.get("focal_point_description") or "").strip()
    elif hasattr(camera_data, "shot_type"):
        shot_type = str(getattr(camera_data, "shot_type", "") or "").strip()
        camera_angle = str(getattr(camera_data, "camera_angle", "") or "").strip()
        focal_point = str(getattr(camera_data, "focal_point_description", "") or "").strip()

    if not shot_type:
        shot_type = str(_extract(scene, "shot_type", "") or "").strip()
    if not camera_angle:
        camera_angle = str(_extract(scene, "camera_angle", "") or "").strip()
    if not focal_point:
        focal_point = str(_extract(scene, "focal_point", "") or "").strip()

    char_state = _extract(scene, "character_state")
    if hasattr(char_state, "model_dump"):
        char_state = char_state.model_dump()
    elif hasattr(char_state, "dict"):
        char_state = char_state.dict()
    elif not isinstance(char_state, dict):
        char_state = None

    env_override = _extract(scene, "environment_state_override")
    transition = str(_extract(scene, "transition_from_previous", "") or "").strip()
    visual_purpose = str(_extract(scene, "visual_purpose", "") or "").strip()
    narr = str(
        _extract(scene, "narration_text", "")
        or _extract(scene, "narration", "")
        or ""
    ).strip()

    # Specialized Preset Branch (e.g. stickman_tactical_explainer, stylized_2d_narrative)
    if resolved_preset is not None and resolved_preset.id != "legacy_default":
        spec_segments: list[str] = []

        # Segment 1: Preset Permanent Anchors & Lighting
        preset_anchors = list(resolved_preset.permanent_prompt_anchors or [])
        art_parts: list[str] = [", ".join(preset_anchors)] if preset_anchors else [f"{resolved_preset.display_name} visual style"]
        lighting_rule = _clean_rule_text(
            resolved_preset.rules.lighting_style
            if resolved_preset.rules and resolved_preset.rules.lighting_style
            else str(art_dir.get("lighting_philosophy", "")).strip()
        )
        if lighting_rule:
            art_parts.append(f"Lighting: {lighting_rule}")
        spec_segments.append(_clean_segment(". ".join(art_parts)))

        # Segment 2: Bound Character (adapted via adapt_character)
        if char_id and char_id in chars_by_id:
            c_entry = chars_by_id[char_id]
            adapted_c = adapt_character(c_entry, resolved_preset)
            char_desc = adapted_c.adapted_description

            if char_state:
                pose = _extract(char_state, "pose_description")
                expr = _extract(char_state, "expression_description")
                action = _extract(char_state, "action_in_progress")
                state_bits: list[str] = []
                if pose:
                    state_bits.append(f"pose: {pose}")
                if expr:
                    state_bits.append(f"expression: {expr}")
                if action:
                    state_bits.append(f"action: {action}")
                if state_bits:
                    char_desc += f". Character state: {', '.join(state_bits)}"
            spec_segments.append(_clean_segment(char_desc))

        # Segment 3: Bound Concept Diagram OR Bound Environment (adapted via adapt_concept / adapt_environment)
        # Check whether this beat needs a concept diagram or is an environmental/narrative moment
        is_diagram_beat = False
        if concept_id and concept_id in concepts_by_id:
            shot_type_lower = shot_type.lower()
            tech_lower = str(_extract(scene, "visual_technique", "") or "").lower()
            trans_lower = transition.lower()
            is_explicit_diagram = (
                shot_type_lower in ("diagram", "cutaway", "cross-section")
                or any(t in tech_lower for t in ("diagram", "schematic", "cutaway", "cross-section", "technical", "infographic"))
                or trans_lower in ("explanatory_cutaway", "surreal_metaphor")
            )
            # If no environment is bound, concept diagram acts as the primary spatial setting
            is_diagram_beat = is_explicit_diagram or not (env_id and env_id in envs_by_id)

        env_additional_negs: list[str] = []
        if concept_id and concept_id in concepts_by_id and is_diagram_beat:
            cp_entry = concepts_by_id[concept_id]
            sliced_elems, foc_exp, foc_ana = slice_concept_for_beat(cp_entry, scene)
            adapted_cp = adapt_concept(
                cp_entry,
                resolved_preset,
                relevant_elements=sliced_elems,
                focused_explanation=foc_exp,
                focused_analogy=foc_ana,
            )
            spec_segments.append(_clean_segment(adapted_cp.adapted_description))
        elif env_id and env_id in envs_by_id:
            e_entry = envs_by_id[env_id]
            sliced_props, incl_layout, local_set, env_negs = slice_environment_for_beat(e_entry, scene)
            light_override = env_override if (env_override and any(w in str(env_override).lower() for w in ("light", "shadow", "dark", "glow", "torch", "candle", "lamp", "sun", "moon"))) else None
            adapted_env = adapt_environment(
                e_entry,
                resolved_preset,
                relevant_props=sliced_props,
                include_spatial_layout=incl_layout,
                local_setting=local_set,
                lighting_override=light_override,
            )
            env_desc = adapted_env.adapted_description
            if env_override and not light_override:
                env_desc += f". Condition: {env_override}"
            spec_segments.append(_clean_segment(env_desc))
            env_additional_negs = env_negs
        elif env_override:
            spec_segments.append(_clean_segment(f"Environment setting: {env_override}"))

        # Segment 4: Bound Props (adapted via adapt_object)
        if obj_ids:
            prop_descs: list[str] = []
            for oid in obj_ids:
                if oid in objs_by_id:
                    adapted_o = adapt_object(objs_by_id[oid], resolved_preset)
                    prop_descs.append(adapted_o.adapted_description)
            if prop_descs:
                spec_segments.append(_clean_segment("; ".join(prop_descs)))

        # Segment 5: Camera Direction & Focal Point
        cam_parts: list[str] = []
        if shot_type and shot_type.lower() not in ("default", ""):
            cam_parts.append(f"{shot_type.replace('_', ' ')} shot")
        if camera_angle and camera_angle.lower() not in ("eye-level", "eye_level", "default", ""):
            cam_parts.append(f"{camera_angle.replace('_', ' ')} angle")
        if cam_parts:
            spec_segments.append(_clean_segment(f"Camera direction: {', '.join(cam_parts)}"))
        if focal_point and focal_point.lower() not in ("default", ""):
            spec_segments.append(_clean_segment(f"Primary focal point: {focal_point}"))

        # Segment 6: Narrative Transition Rationale & Visual Action
        context_parts: list[str] = []
        if transition == "explanatory_cutaway":
            context_parts.append("Framing context: explanatory cutaway transition")
        elif transition == "surreal_metaphor":
            context_parts.append("Framing context: surreal visual metaphor")

        visual_desc = str(_extract(scene, "visual_description", "") or "").strip()
        action_text = visual_desc or visual_purpose
        if action_text and len(action_text.split()) >= 3:
            context_parts.append(f"Visual moment: {action_text}")

        intent_val = str(_extract(scene, "visual_intent", "") or "").strip().lower()
        if intent_val == "human_experience" and not char_id and not any("human" in c.lower() for c in context_parts):
            context_parts.append("Framing context: human reaction and community experience")
        elif intent_val == "narrative_resolution" and not any("resolution" in c.lower() for c in context_parts):
            context_parts.append("Framing context: narrative resolution")

        if context_parts:
            spec_segments.append(_clean_segment(". ".join(context_parts)))

        # Segment 7: Color Palette & Style Drawing Rules
        palette_rule = _clean_rule_text(
            resolved_preset.rules.color_and_shading
            if resolved_preset.rules and resolved_preset.rules.color_and_shading
            else ""
        )
        palette_list = resolved_preset.default_palette_hex or art_dir.get("primary_palette_hex") or []
        palette_parts: list[str] = []
        if palette_list:
            palette_parts.append(", ".join(str(p) for p in palette_list))
        if palette_rule:
            palette_parts.append(palette_rule)
        if palette_parts:
            spec_segments.append(_clean_segment(f"Color palette: {'; '.join(palette_parts)}"))

        # Segment 8: Negative Constraints & Avoidances
        is_standalone_concept = bool(concept_id and not char_id)
        neg_anchors: list[str] = list(resolved_preset.negative_prompt_anchors or [])

        # Add global world/era constraints (applies to all scenes, including when character_id is None)
        world_constraints = _extract_world_negative_constraints(
            art_dir=art_dir,
            creative_direction=creative_direction,
            is_standalone_concept=is_standalone_concept,
        )
        for wc in world_constraints:
            clean_wc = wc.strip()
            lower_wc = clean_wc.lower()
            norm_wc = re.sub(r"^(?:no|avoid|never|do not|don't)\s+", "", lower_wc)
            existing_norms = {re.sub(r"^(?:no|avoid|never|do not|don't)\s+", "", a.lower()) for a in neg_anchors}
            if norm_wc not in existing_norms:
                neg_anchors.append(clean_wc)

        # Character forbidden mutations (strictly character-specific: applied ONLY when this character is bound)
        if char_id and char_id in chars_by_id:
            c_entry = chars_by_id[char_id]
            c_mutations = _extract(c_entry, "forbidden_mutations") or []
            for mut in c_mutations:
                m_str = str(mut).strip()
                if m_str and m_str not in neg_anchors:
                    neg_anchors.append(m_str)

        # Standalone conceptual explanation beats: strictly forbid physical bedroom/sleeper elements ONLY when bedroom context exists
        has_bedroom = _has_bedroom_context(
            entities=entities,
            creative_direction=creative_direction,
            video_topic=video_topic,
            art_dir=art_dir,
        )
        if is_standalone_concept and has_bedroom:
            if not any("bedroom" in r.lower() for r in neg_anchors):
                neg_anchors.append("no bedroom setting")

        # Scarcity / famine / crop failure targeted negative constraints
        beat_full_text = f"{narr} {visual_desc} {focal_point} {action_text}"
        scarcity_negatives = extract_scarcity_negative_constraints(beat_full_text)
        for sn in scarcity_negatives:
            clean_sn = sn.strip()
            lower_sn = clean_sn.lower()
            norm_sn = re.sub(r"^(?:no|avoid|never|do not|don't)\s+", "", lower_sn)
            existing_norms = {re.sub(r"^(?:no|avoid|never|do not|don't)\s+", "", a.lower()) for a in neg_anchors}
            if norm_sn not in existing_norms:
                neg_anchors.append(clean_sn)

        for en in env_additional_negs:
            clean_en = en.strip()
            lower_en = clean_en.lower()
            norm_en = re.sub(r"^(?:no|avoid|never|do not|don't)\s+", "", lower_en)
            existing_norms = {re.sub(r"^(?:no|avoid|never|do not|don't)\s+", "", a.lower()) for a in neg_anchors}
            if norm_en not in existing_norms:
                neg_anchors.append(clean_en)

        if neg_anchors:
            spec_segments.append(_clean_segment(f"Avoid: {', '.join(neg_anchors)}"))

        # Segment 9: Fixed Composition Rules
        spec_segments.append(_clean_segment(COMPOSITION_RULES))

        spec_joined = " ".join(spec_segments)
        return re.sub(r"\s+", " ", spec_joined).strip()

    # Legacy / Unconfigured Default Path (Exact historical implementation)
    segments: list[str] = []

    is_standalone_concept = bool(concept_id and not char_id)

    # Segment 1: Art Direction & Style Engine
    style_family = str(art_dir.get("style_family", "cinematic")).strip()
    visual_genre = str(art_dir.get("visual_genre", "photorealistic_cinematic")).replace("_", " ").strip()
    texture = str(art_dir.get("rendering_texture", "")).strip()
    lighting_phil = str(art_dir.get("lighting_philosophy", "")).strip()

    art_parts: list[str] = [f"{visual_genre.capitalize()} visual style"]
    if texture:
        if is_standalone_concept:
            # Filter out physical scene textures (e.g. bedding, duvet, mattress, pillows, cotton weave, tousled hair, clothing, skin shading)
            physical_terms = (
                "cotton weave",
                "bedding",
                "rumpled",
                "bed",
                "duvet",
                "pillow",
                "mattress",
                "nightstand",
                "furniture",
                "tousled hair",
                "hair",
                "clothing",
                "shirt",
                "pajama",
                "fabric",
                "skin shading",
            )
            parts = [p.strip() for p in texture.split(",") if p.strip()]
            kept_textures = []
            for part in parts:
                low = part.lower().rstrip(".")
                if any(t in low for t in physical_terms):
                    continue
                clean_p = re.sub(r"^(and\s+)", "", part.strip(), flags=re.IGNORECASE).strip().rstrip(".")
                if clean_p:
                    kept_textures.append(clean_p)
            if kept_textures:
                art_parts.append(", ".join(kept_textures))
        else:
            art_parts.append(texture)
    segments.append(_clean_segment(", ".join(art_parts)))

    # Segment 2: Bound Character (ONLY if character_id is explicitly bound)
    if char_id and char_id in chars_by_id:
        c_entry = chars_by_id[char_id]
        c_name = _extract(c_entry, "name", "Protagonist")
        c_summary = _extract(c_entry, "visual_anchor") or _extract(c_entry, "visual_summary", "")
        c_clothing = _extract(c_entry, "apparel_anchor") or _extract(c_entry, "clothing_and_style", "")

        char_desc = f"Subject: {c_name}, {c_summary}"
        if c_clothing:
            char_desc += f", wearing {c_clothing}"

        if char_state:
            pose = _extract(char_state, "pose_description")
            expr = _extract(char_state, "expression_description")
            action = _extract(char_state, "action_in_progress")
            state_bits: list[str] = []
            if pose:
                state_bits.append(f"pose: {pose}")
            if expr:
                state_bits.append(f"expression: {expr}")
            if action:
                state_bits.append(f"action: {action}")
            if state_bits:
                char_desc += f". Character state: {', '.join(state_bits)}"
        segments.append(_clean_segment(char_desc))

    # Segment 3: Bound Concept Diagram OR Bound Environment
    is_diagram_beat = False
    env_additional_negs: list[str] = []
    if concept_id and concept_id in concepts_by_id:
        shot_type_lower = shot_type.lower()
        tech_lower = str(_extract(scene, "visual_technique", "") or "").lower()
        trans_lower = transition.lower()
        is_explicit_diagram = (
            shot_type_lower in ("diagram", "cutaway", "cross-section")
            or any(t in tech_lower for t in ("diagram", "schematic", "cutaway", "cross-section", "technical", "infographic"))
            or trans_lower in ("explanatory_cutaway", "surreal_metaphor")
        )
        is_diagram_beat = is_explicit_diagram or not (env_id and env_id in envs_by_id)

    if concept_id and concept_id in concepts_by_id and is_diagram_beat:
        cp_entry = concepts_by_id[concept_id]
        cp_name = _extract(cp_entry, "name", "")
        sliced_elems, foc_exp, foc_ana = slice_concept_for_beat(cp_entry, scene)
        cp_style = foc_ana or _extract(cp_entry, "visual_analogy") or _extract(cp_entry, "diagram_style", "")
        cp_rules = foc_exp or _extract(cp_entry, "concept_explained") or _extract(cp_entry, "labeling_rules", "")
        if is_standalone_concept:
            cp_style = _clean_conceptual_analogy(str(cp_style))
            cp_rules = _clean_conceptual_styling(str(cp_rules))
        cpt_desc = f"Visual diagram concept: {cp_name}, {cp_style}"
        if sliced_elems:
            cpt_desc += f", graphical elements: {', '.join(str(e) for e in sliced_elems)}"
        if cp_rules:
            cpt_desc += f". Styling: {cp_rules}"
        segments.append(_clean_segment(cpt_desc))
    elif env_id and env_id in envs_by_id:
        e_entry = envs_by_id[env_id]
        e_name = _extract(e_entry, "name", "")
        e_desc = _extract(e_entry, "spatial_layout") or _extract(e_entry, "description", "")
        e_light = _extract(e_entry, "lighting_setup") or _extract(e_entry, "lighting", "")
        sliced_props, incl_layout, local_set, env_negs = slice_environment_for_beat(e_entry, scene)

        setting_name = local_set or e_name
        setting_desc = e_desc if (incl_layout and not local_set) else ""
        env_desc = f"Environment setting: {setting_name}"
        if setting_desc:
            env_desc += f", {setting_desc}"
        if sliced_props:
            env_desc += f". Persistent props: {'; '.join(str(p) for p in sliced_props)}"
        if env_override:
            env_desc += f". Condition: {env_override}"
        if e_light and e_light.lower() not in lighting_phil.lower():
            env_desc += f". Lighting: {e_light}"
        segments.append(_clean_segment(env_desc))
        env_additional_negs = env_negs
    elif env_override:
        segments.append(_clean_segment(f"Environment setting: {env_override}"))
    elif lighting_phil:
        segments.append(_clean_segment(f"Lighting: {lighting_phil}"))

    # Segment 4: Bound Props (ONLY if explicitly listed in object_ids)
    if obj_ids:
        prop_descs: list[str] = []
        for oid in obj_ids:
            if oid in objs_by_id:
                o_entry = objs_by_id[oid]
                o_name = _extract(o_entry, "name", "")
                o_det = _extract(o_entry, "visual_description") or _extract(o_entry, "visual_details", "")
                prop_descs.append(f"{o_name} ({o_det})")
        if prop_descs:
            segments.append(_clean_segment(f"Visible props: {'; '.join(prop_descs)}"))

    # Segment 5: Camera Direction & Focal Point
    cam_parts: list[str] = []
    if shot_type and shot_type.lower() not in ("default", ""):
        cam_parts.append(f"{shot_type.replace('_', ' ')} shot")
    if camera_angle and camera_angle.lower() not in ("eye-level", "eye_level", "default", ""):
        cam_parts.append(f"{camera_angle.replace('_', ' ')} angle")
    if cam_parts:
        segments.append(_clean_segment(f"Camera direction: {', '.join(cam_parts)}"))
    if focal_point and focal_point.lower() not in ("default", ""):
        segments.append(_clean_segment(f"Primary focal point: {focal_point}"))

    # Segment 6: Narrative Transition Rationale & Visual Action
    context_parts: list[str] = []
    if transition == "explanatory_cutaway":
        context_parts.append("Framing context: explanatory cutaway transition")
    elif transition == "surreal_metaphor":
        context_parts.append("Framing context: surreal visual metaphor")

    visual_desc = str(_extract(scene, "visual_description", "") or "").strip()
    action_text = visual_desc or visual_purpose
    if action_text and len(action_text.split()) >= 3:
        context_parts.append(f"Visual moment: {action_text}")

    if context_parts:
        segments.append(_clean_segment(". ".join(context_parts)))

    # Segment 7: Color Palette Harmony
    palette = art_dir.get("primary_palette_hex") or []
    if palette:
        palette_str = ", ".join(str(p) for p in palette)
        segments.append(_clean_segment(f"Color palette: {palette_str}"))

    # Segment 8: Continuity Rules and Negative Constraints
    world_constraints = _extract_world_negative_constraints(
        art_dir=art_dir,
        creative_direction=creative_direction,
        is_standalone_concept=is_standalone_concept,
    )

    continuity_rules: list[str] = []
    negative_rules: list[str] = []

    prohibitive_prefixes = ("no ", "never ", "do not ", "don't ", "avoid ", "without ", "stop ")
    affirmative_prefixes = ("keep ", "maintain ", "ensure ", "always ")

    for item in world_constraints:
        s = str(item).strip()
        if not s:
            continue
        lower = s.lower()
        if any(lower.startswith(p) for p in affirmative_prefixes):
            # In standalone conceptual explanation beats, omit continuity rules mandating sleeper or bedroom
            if concept_id and not char_id and any(term in lower for term in ("sleeper", "character", "bedroom", "person")):
                continue
            if s not in continuity_rules:
                continuity_rules.append(s)
        else:
            clean_neg = s if any(lower.startswith(p) for p in prohibitive_prefixes) else f"no {s}"
            if clean_neg not in negative_rules:
                negative_rules.append(clean_neg)

    # Character forbidden mutations (only when a character is actively bound)
    if char_id and char_id in chars_by_id:
        c_entry = chars_by_id[char_id]
        c_mutations = _extract(c_entry, "forbidden_mutations") or []
        for mut in c_mutations:
            m_str = str(mut).strip()
            if not m_str:
                continue
            lower_m = m_str.lower()
            clean_m = m_str if any(lower_m.startswith(p) for p in prohibitive_prefixes) else f"no {m_str}"
            if clean_m not in negative_rules:
                negative_rules.append(clean_m)

    # Standalone conceptual explanation beats: strictly forbid physical bedroom/sleeper elements ONLY when bedroom context exists
    has_bedroom = _has_bedroom_context(
        entities=entities,
        creative_direction=creative_direction,
        video_topic=video_topic,
        art_dir=art_dir,
    )
    if concept_id and not char_id and has_bedroom:
        if not any("bedroom" in r.lower() for r in negative_rules):
            negative_rules.append("no bedroom setting")

    # Scarcity / famine / crop failure targeted negative constraints
    beat_full_text = f"{narr} {visual_desc} {focal_point} {action_text}"
    scarcity_negatives = extract_scarcity_negative_constraints(beat_full_text)
    for sn in scarcity_negatives:
        clean_sn = sn if any(sn.lower().startswith(p) for p in prohibitive_prefixes) else f"no {sn}"
        if clean_sn not in negative_rules:
            negative_rules.append(clean_sn)

    for en in env_additional_negs:
        clean_en = en if any(en.lower().startswith(p) for p in prohibitive_prefixes) else f"no {en}"
        if clean_en not in negative_rules:
            negative_rules.append(clean_en)

    if continuity_rules:
        segments.append(_clean_segment(f"Continuity rules: {', '.join(continuity_rules)}"))
    if negative_rules:
        segments.append(_clean_segment(f"Negative constraints: {', '.join(negative_rules)}"))

    # Segment 9: Fixed Composition Rules
    segments.append(_clean_segment(COMPOSITION_RULES))

    joined = " ".join(segments)
    return re.sub(r"\s+", " ", joined).strip()


def compose_final_prompt(
    scene: Any,
    subject_anchor: str | None = None,
    visual_style: str | None = None,
    *,
    environment_anchor: str | None = None,
    video_topic: str | None = None,
    video_title: str | None = None,
    prev_scene: Any | None = None,
    next_scene: Any | None = None,
    video_bible: dict[str, Any] | Any | None = None,
    creative_direction: dict[str, Any] | Any | None = None,
    preset: VisualStylePreset | dict[str, Any] | str | None = None,
) -> str:
    """Compose the final image prompt for a scene with rich visual context.

    When a Video Bible is provided (8-step pipeline), deterministically composes
    the prompt using bound entities, art direction, and camera instructions,
    adapting entities to the active visual style preset if present.
    When Video Bible is absent (legacy 6-step videos), preserves legacy composition.
    """
    resolved_preset = _resolve_preset(preset, context=scene)

    # Check for Video Bible presence (explicit or attached to scene dict)
    resolved_vb = video_bible
    if resolved_vb is None and isinstance(scene, dict):
        resolved_vb = scene.get("video_bible") or scene.get("_video_bible")

    if resolved_vb:
        connected_result = _compose_connected_final_prompt(
            scene=scene,
            video_bible=resolved_vb,
            creative_direction=creative_direction,
            preset=resolved_preset,
            video_topic=video_topic,
        )
        if connected_result:
            return connected_result
    if isinstance(scene, dict):
        img_prompt = scene.get("image_prompt", "")
        uses_anchor = bool(scene.get("uses_anchor", False))
        shot_type = str(scene.get("shot_type", "") or "").strip()
        camera_angle = str(scene.get("camera_angle", "") or "").strip()
        focal_point = str(scene.get("focal_point", "") or "").strip()
        visual_technique = str(scene.get("visual_technique", "") or "").strip()
        visual_purpose = str(scene.get("visual_purpose", "") or "").strip()
        narration = str(scene.get("narration", "") or "").strip()
        environment = str(scene.get("environment", "") or "").strip()
    else:
        img_prompt = getattr(scene, "image_prompt", "")
        uses_anchor = bool(getattr(scene, "uses_anchor", False))
        shot_type = str(getattr(scene, "shot_type", "") or "").strip()
        camera_angle = str(getattr(scene, "camera_angle", "") or "").strip()
        focal_point = str(getattr(scene, "focal_point", "") or "").strip()
        visual_technique = str(getattr(scene, "visual_technique", "") or "").strip()
        visual_purpose = str(getattr(scene, "visual_purpose", "") or "").strip()
        narration = str(getattr(scene, "narration", "") or "").strip()
        environment = str(getattr(scene, "environment", "") or "").strip()

    segments: list[str] = []

    # 1. Scene image prompt (Subject & primary action)
    cleaned_img = _clean_segment(str(img_prompt or ""))
    if cleaned_img:
        segments.append(cleaned_img)

    # 2. Subject anchor (only if uses_anchor is True and anchor is provided)
    if uses_anchor and subject_anchor:
        cleaned_anchor = _clean_segment(str(subject_anchor).strip())
        if cleaned_anchor:
            segments.append(cleaned_anchor)

    # 3. Environment & Background Setting
    effective_env = environment or (environment_anchor or "").strip()
    if effective_env:
        # Avoid redundant phrasing if the environment is already fully in cleaned_img
        if effective_env.lower() not in cleaned_img.lower():
            prefix = "Environment" if environment else "Environment setting"
            segments.append(_clean_segment(f"{prefix}: {effective_env}"))

    # 4. Narration and Visual Concept Context
    if visual_purpose:
        if visual_purpose.lower() not in cleaned_img.lower():
            segments.append(_clean_segment(f"Visual concept: {visual_purpose}"))
    elif narration and len(narration.split()) >= 3:
        if narration.lower() not in cleaned_img.lower():
            segments.append(_clean_segment(f"Narration context: {narration}"))

    # 5. Camera framing and perspective direction (shot_type, camera_angle)
    camera_parts: list[str] = []
    if shot_type and shot_type.lower() not in ("default", ""):
        camera_parts.append(f"{shot_type} shot")
    if camera_angle and camera_angle.lower() not in ("eye-level", "default", ""):
        angle_clean = camera_angle.strip()
        if not angle_clean.lower().endswith(("angle", "perspective", "view")):
            angle_clean += " angle"
        camera_parts.append(angle_clean)
    if camera_parts:
        segments.append(_clean_segment(f"Camera direction: {', '.join(camera_parts)}"))

    # 6. Primary visual focal point
    if focal_point and focal_point.lower() not in ("default", ""):
        if focal_point.lower() not in cleaned_img.lower():
            segments.append(_clean_segment(f"Primary focal point: {focal_point}"))

    # 7. Visual technique
    if visual_technique and visual_technique.lower() not in ("cinematic-photo", "default", ""):
        if visual_technique.lower() not in cleaned_img.lower():
            segments.append(_clean_segment(f"Visual technique: {visual_technique}"))

    # 8. Adaptive Visual Style
    if resolved_preset is not None and resolved_preset.id != "legacy_default":
        style_text = ", ".join(resolved_preset.permanent_prompt_anchors)
        cleaned_style = _clean_segment(style_text)
        if cleaned_style:
            segments.append(cleaned_style)
    elif visual_style:
        adapted_style = adapt_visual_style(
            visual_style=visual_style,
            environment=effective_env,
            shot_type=shot_type,
            visual_technique=visual_technique,
        )
        cleaned_style = _clean_segment(adapted_style)
        if cleaned_style:
            segments.append(cleaned_style)

    # 9. Fixed Composition Rules & Negative Constraints
    if (
        resolved_preset is not None
        and resolved_preset.id != "legacy_default"
        and resolved_preset.negative_prompt_anchors
    ):
        neg_str = f"Avoid: {', '.join(resolved_preset.negative_prompt_anchors)}"
        segments.append(_clean_segment(neg_str))

    cleaned_rules = _clean_segment(COMPOSITION_RULES)
    if cleaned_rules:
        segments.append(cleaned_rules)

    joined = " ".join(segments)
    return re.sub(r"\s+", " ", joined).strip()


def recompute_final_prompts(
    storyboard: dict[str, Any],
    visual_style: str | None = None,
    video_bible: dict[str, Any] | Any | None = None,
    creative_direction: dict[str, Any] | Any | None = None,
    preset: VisualStylePreset | dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Rebuild every scene's final_prompt from its component parts deterministically.

    Can be invoked on the root step output (e.g. step.output) or directly
    on the inner storyboard dict. Updates scenes in-place and returns the dict.
    Preserves all semantic storyboard data (narration, sequence, timing, actions, bindings).
    """
    target = (
        storyboard.get("storyboard")
        if isinstance(storyboard.get("storyboard"), dict)
        else storyboard
    )

    resolved_preset = _resolve_preset(preset, context=target)
    if resolved_preset is None and isinstance(storyboard, dict):
        resolved_preset = _resolve_preset(preset, context=storyboard)

    if resolved_preset is not None:
        target["visual_style_id"] = resolved_preset.id
        target["visual_style_snapshot"] = (
            resolved_preset.model_dump()
            if hasattr(resolved_preset, "model_dump")
            else dict(resolved_preset)
        )
        if visual_style is None:
            if resolved_preset.id != "legacy_default":
                anchors_str = ", ".join(resolved_preset.permanent_prompt_anchors)
                rules_str = (
                    f"Linework: {resolved_preset.rules.linework_and_outlines}. "
                    f"Shading: {resolved_preset.rules.color_and_shading}."
                )
                style = f"{resolved_preset.display_name}: {anchors_str}. {rules_str}"
            else:
                style = target.get("visual_style") or ""
        else:
            style = visual_style
        target["visual_style"] = style
    else:
        style = visual_style or target.get("visual_style") or ""

    anchor = target.get("subject_anchor") or ""
    env_anchor = target.get("environment_anchor") or ""
    vb = video_bible or target.get("video_bible")
    cd = creative_direction or target.get("creative_direction")
    topic = target.get("topic") or target.get("video_topic") or ""
    if not topic and isinstance(vb, dict):
        topic = vb.get("topic_summary") or ""

    scenes = target.get("scenes") or []
    for idx, s in enumerate(scenes):
        prev_s = scenes[idx - 1] if idx > 0 else None
        next_s = scenes[idx + 1] if idx + 1 < len(scenes) else None
        scene_prompt = compose_final_prompt(
            scene=s,
            subject_anchor=anchor,
            visual_style=style,
            environment_anchor=env_anchor,
            video_topic=topic,
            prev_scene=prev_s,
            next_scene=next_s,
            video_bible=vb,
            creative_direction=cd,
            preset=resolved_preset,
        )
        s["final_prompt"] = scene_prompt

        beats = s.get("visual_beats") or []
        for b in beats:
            if isinstance(b, dict):
                b_ctx = {**s, **b}
                if "narration" in s and not b_ctx.get("scene_narration"):
                    b_ctx["scene_narration"] = s.get("narration")
                b["final_prompt"] = compose_final_prompt(
                    scene=b_ctx,
                    subject_anchor=anchor,
                    visual_style=style,
                    environment_anchor=env_anchor,
                    video_topic=topic,
                    prev_scene=prev_s,
                    next_scene=next_s,
                    video_bible=vb,
                    creative_direction=cd,
                    preset=resolved_preset,
                )

    flat_beats = target.get("beats") or []
    for fb in flat_beats:
        if isinstance(fb, dict):
            sc_idx = fb.get("scene_index", 1)
            matching_sc = next((sc for sc in scenes if sc.get("index") == sc_idx), {})
            fb_ctx = {**matching_sc, **fb}
            if "narration" in matching_sc and not fb_ctx.get("scene_narration"):
                fb_ctx["scene_narration"] = matching_sc.get("narration")
            fb["final_prompt"] = compose_final_prompt(
                scene=fb_ctx,
                subject_anchor=anchor,
                visual_style=style,
                environment_anchor=env_anchor,
                video_topic=topic,
                video_bible=vb,
                creative_direction=cd,
                preset=resolved_preset,
            )

    return storyboard
