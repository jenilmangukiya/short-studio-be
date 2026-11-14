"""Prompt composition utilities for Shorts Studio.

Combines scene-specific image prompts, recurring subject anchors, visual styles,
environment settings, camera directions, focal points, and fixed vertical framing rules
into a unified final prompt for image generation.
"""

from __future__ import annotations

import re
from typing import Any

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


def _compose_connected_final_prompt(
    scene: Any,
    video_bible: dict[str, Any] | Any,
    creative_direction: dict[str, Any] | Any | None = None,
) -> str:
    """Compose an image prompt deterministically from Video Bible entities and directorial intent.

    Applies:
    - Global Art Direction (visual genre, lighting philosophy, rendering texture).
    - ONLY the entities explicitly bound to this specific beat (character, environment, props, concept).
    - Camera direction, framing, and primary visual focal point.
    - Narrative transition rationale (e.g. cutaway framing).
    - Art direction negative anchors and vertical 9:16 safe-zone rules.
    """
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

    segments: list[str] = []

    # Segment 1: Art Direction & Style Engine
    style_family = str(art_dir.get("style_family", "cinematic")).strip()
    visual_genre = str(art_dir.get("visual_genre", "photorealistic_cinematic")).replace("_", " ").strip()
    texture = str(art_dir.get("rendering_texture", "")).strip()
    lighting_phil = str(art_dir.get("lighting_philosophy", "")).strip()

    art_parts: list[str] = [f"{visual_genre.capitalize()} visual style"]
    if texture:
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
        cpt_desc = f"Visual diagram concept: {cp_name}, {cp_style}"
        if cp_rules:
            cpt_desc += f". Styling: {cp_rules}"
        segments.append(_clean_segment(cpt_desc))
    elif env_id and env_id in envs_by_id:
        e_entry = envs_by_id[env_id]
        e_name = _extract(e_entry, "name", "")
        e_desc = _extract(e_entry, "spatial_layout") or _extract(e_entry, "description", "")
        e_light = _extract(e_entry, "lighting_setup") or _extract(e_entry, "lighting", "")

        env_desc = f"Environment setting: {e_name}, {e_desc}"
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

    # Segment 6: Narrative Transition Rationale
    if transition == "explanatory_cutaway":
        segments.append(_clean_segment("Framing context: explanatory cutaway transition"))
    elif transition == "surreal_metaphor":
        segments.append(_clean_segment("Framing context: surreal visual metaphor"))
    elif visual_purpose and len(visual_purpose.split()) >= 3:
        segments.append(_clean_segment(f"Visual concept: {visual_purpose}"))

    # Segment 7: Color Palette Harmony
    palette = art_dir.get("primary_palette_hex") or []
    if palette:
        palette_str = ", ".join(str(p) for p in palette)
        segments.append(_clean_segment(f"Color palette: {palette_str}"))

    # Segment 8: Negative Constraints
    neg_anchors = list(art_dir.get("negative_anchors") or [])
    if not neg_anchors and art_dir.get("negative_prompt_tokens"):
        neg_anchors = [str(art_dir.get("negative_prompt_tokens"))]
    if concept_id and not any("bedroom" in str(n).lower() for n in neg_anchors):
        neg_anchors.append("bedroom setting")
    if neg_anchors:
        segments.append(_clean_segment(f"Negative constraints: no {', no '.join(neg_anchors)}"))

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
) -> str:
    """Compose the final image prompt for a scene with rich visual context.

    When a Video Bible is provided (8-step pipeline), deterministically composes
    the prompt using bound entities, art direction, and camera instructions.
    When Video Bible is absent (legacy 6-step videos), preserves legacy composition.
    """
    # Check for Video Bible presence (explicit or attached to scene dict)
    resolved_vb = video_bible
    if resolved_vb is None and isinstance(scene, dict):
        resolved_vb = scene.get("video_bible") or scene.get("_video_bible")

    if resolved_vb:
        connected_result = _compose_connected_final_prompt(
            scene=scene,
            video_bible=resolved_vb,
            creative_direction=creative_direction,
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
    if visual_style:
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
    cleaned_rules = _clean_segment(COMPOSITION_RULES)
    if cleaned_rules:
        segments.append(cleaned_rules)

    joined = " ".join(segments)
    return re.sub(r"\s+", " ", joined).strip()


def recompute_final_prompts(
    storyboard: dict[str, Any],
    visual_style: str | None = None,
    video_bible: dict[str, Any] | Any | None = None,
) -> dict[str, Any]:
    """Rebuild every scene's final_prompt from its component parts.

    Can be invoked on the root step output (e.g. step.output) or directly
    on the inner storyboard dict. Updates scenes in-place and returns the dict.
    """
    target = (
        storyboard.get("storyboard")
        if isinstance(storyboard.get("storyboard"), dict)
        else storyboard
    )

    style = visual_style or target.get("visual_style") or ""
    anchor = target.get("subject_anchor") or ""
    env_anchor = target.get("environment_anchor") or ""
    vb = video_bible or target.get("video_bible")

    scenes = target.get("scenes") or []
    for idx, s in enumerate(scenes):
        prev_s = scenes[idx - 1] if idx > 0 else None
        next_s = scenes[idx + 1] if idx + 1 < len(scenes) else None
        s["final_prompt"] = compose_final_prompt(
            scene=s,
            subject_anchor=anchor,
            visual_style=style,
            environment_anchor=env_anchor,
            prev_scene=prev_s,
            next_scene=next_s,
            video_bible=vb,
        )

    return storyboard
