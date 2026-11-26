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
) -> str:
    """Compose an image prompt deterministically from Video Bible entities and directorial intent.

    Applies:
    - Global Art Direction (visual genre, lighting philosophy, rendering texture) or visual style preset.
    - ONLY the entities explicitly bound to this specific beat (character, environment, props, concept),
      adapted to the visual style preset when active.
    - Camera direction, framing, and primary visual focal point.
    - Narrative transition rationale (e.g. cutaway framing).
    - Art direction negative anchors, preset negative anchors, and vertical 9:16 safe-zone rules.
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

    # Enforce strict character prohibition when character_count is 'none' or no characters in catalog
    cd_dict = (
        creative_direction.model_dump()
        if hasattr(creative_direction, "model_dump")
        else dict(creative_direction) if creative_direction else {}
    )
    protagonist_cfg = cd_dict.get("protagonist_config") or {}
    if hasattr(protagonist_cfg, "model_dump"):
        protagonist_cfg = protagonist_cfg.model_dump()
    expected_char_count = str(protagonist_cfg.get("character_count", "")).lower()
    if expected_char_count == "none" or not chars_by_id:
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
        if concept_id and concept_id in concepts_by_id:
            cp_entry = concepts_by_id[concept_id]
            adapted_cp = adapt_concept(cp_entry, resolved_preset)
            spec_segments.append(_clean_segment(adapted_cp.adapted_description))
        elif env_id and env_id in envs_by_id:
            e_entry = envs_by_id[env_id]
            adapted_env = adapt_environment(e_entry, resolved_preset)
            env_desc = adapted_env.adapted_description
            if env_override:
                env_desc += f". Condition: {env_override}"
            spec_segments.append(_clean_segment(env_desc))
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

        # Standalone conceptual explanation beats: strictly forbid physical bedroom/sleeper elements
        if is_standalone_concept:
            if not any("bedroom" in r.lower() for r in neg_anchors):
                neg_anchors.append("no bedroom setting")

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
    if concept_id and concept_id in concepts_by_id:
        cp_entry = concepts_by_id[concept_id]
        cp_name = _extract(cp_entry, "name", "")
        cp_style = _extract(cp_entry, "visual_analogy") or _extract(cp_entry, "diagram_style", "")
        cp_rules = _extract(cp_entry, "concept_explained") or _extract(cp_entry, "labeling_rules", "")
        if is_standalone_concept:
            cp_style = _clean_conceptual_analogy(str(cp_style))
            cp_rules = _clean_conceptual_styling(str(cp_rules))
        cpt_desc = f"Visual diagram concept: {cp_name}, {cp_style}"
        if cp_rules:
            cpt_desc += f". Styling: {cp_rules}"
        segments.append(_clean_segment(cpt_desc))
    elif env_id and env_id in envs_by_id:
        e_entry = envs_by_id[env_id]
        e_name = _extract(e_entry, "name", "")
        e_desc = _extract(e_entry, "spatial_layout") or _extract(e_entry, "description", "")
        e_light = _extract(e_entry, "lighting_setup") or _extract(e_entry, "lighting", "")
        e_props = _extract(e_entry, "persistent_props") or []

        env_desc = f"Environment setting: {e_name}, {e_desc}"
        if e_props:
            env_desc += f". Persistent props: {'; '.join(str(p) for p in e_props)}"
        if env_override:
            env_desc += f". Condition: {env_override}"
        if e_light and e_light.lower() not in lighting_phil.lower():
            env_desc += f". Lighting: {e_light}"
        segments.append(_clean_segment(env_desc))
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

    # Standalone conceptual explanation beats: strictly forbid physical bedroom/sleeper elements
    if concept_id and not char_id:
        if not any("bedroom" in r.lower() for r in negative_rules):
            negative_rules.append("no bedroom setting")

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

    scenes = target.get("scenes") or []
    for idx, s in enumerate(scenes):
        prev_s = scenes[idx - 1] if idx > 0 else None
        next_s = scenes[idx + 1] if idx + 1 < len(scenes) else None
        scene_prompt = compose_final_prompt(
            scene=s,
            subject_anchor=anchor,
            visual_style=style,
            environment_anchor=env_anchor,
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
                b["final_prompt"] = compose_final_prompt(
                    scene=b_ctx,
                    subject_anchor=anchor,
                    visual_style=style,
                    environment_anchor=env_anchor,
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
            fb["final_prompt"] = compose_final_prompt(
                scene=fb_ctx,
                subject_anchor=anchor,
                visual_style=style,
                environment_anchor=env_anchor,
                video_bible=vb,
                creative_direction=cd,
                preset=resolved_preset,
            )

    return storyboard
