"""Continuity and story progression validation for connected storyboards.

Validates that storyboard scenes adhere to Video Bible entities, Creative Direction
guidance, intentional environment transitions, visual diversity, and camera progression.
"""

from __future__ import annotations

import re
from typing import Any

from app.schemas import CreativeDirectionOutput, VideoBibleOutput


def validate_connected_storyboard(
    scenes: list[dict[str, Any] | Any],
    video_bible: VideoBibleOutput | dict[str, Any] | None,
    creative_direction: CreativeDirectionOutput | dict[str, Any] | None = None,
) -> list[str]:
    """Validate connected storyboard continuity and return a list of actionable warnings.

    Performs deterministic structural validation:
    1. Entity ID existence (characters, environments, objects, concepts) against Video Bible.
    2. Character usage consistency with Creative Direction protagonist configuration.
    3. Motivated environment transitions (no unannounced location jumps on same-scene cuts).
    4. Camera framing diversity and consecutive repetition prevention.
    5. Standalone concept integrity (diagram beats should not reference physical bedroom props).
    """
    warnings: list[str] = []
    if not scenes:
        warnings.append("Storyboard contains no scenes or visual beats.")
        return warnings

    if not video_bible:
        # Legacy storyboard without Video Bible skips entity catalog checks
        return warnings

    # 1. Normalize Video Bible data
    vb_dict = (
        video_bible.model_dump()
        if hasattr(video_bible, "model_dump")
        else dict(video_bible)
    )
    entities = vb_dict.get("entities") or {}
    if hasattr(entities, "model_dump"):
        entities = entities.model_dump()

    known_character_ids = {
        str(c.get("id") if isinstance(c, dict) else getattr(c, "id", "")).strip()
        for c in (entities.get("characters") or [])
    } - {""}
    known_environment_ids = {
        str(e.get("id") if isinstance(e, dict) else getattr(e, "id", "")).strip()
        for e in (entities.get("environments") or [])
    } - {""}
    known_object_ids = {
        str(o.get("id") if isinstance(o, dict) else getattr(o, "id", "")).strip()
        for o in (entities.get("objects") or entities.get("objects_and_props") or [])
    } - {""}
    known_concept_ids = {
        str(cp.get("id") if isinstance(cp, dict) else getattr(cp, "id", "")).strip()
        for cp in (entities.get("concepts") or entities.get("concepts_and_diagrams") or [])
    } - {""}

    # 2. Normalize Creative Direction data if present
    cd_dict = None
    if creative_direction:
        cd_dict = (
            creative_direction.model_dump()
            if hasattr(creative_direction, "model_dump")
            else dict(creative_direction)
        )
    protagonist_cfg = cd_dict.get("protagonist_config") if cd_dict else {}
    if hasattr(protagonist_cfg, "model_dump"):
        protagonist_cfg = protagonist_cfg.model_dump()
    human_presence = (
        protagonist_cfg.get("human_presence", "none")
        if isinstance(protagonist_cfg, dict)
        else "none"
    )

    character_occurrences = 0
    shot_types_used: set[str] = set()
    consecutive_shot_repeats = 1
    last_shot_type: str | None = None
    last_environment_id: str | None = None
    consecutive_diagrams = 0
    distinct_diagram_stages = 0
    redundant_diagrams = 0
    prev_diagram_info: tuple[str, str, str] | None = None

    for idx, sc in enumerate(scenes, start=1):
        def _get(field: str, default: Any = None) -> Any:
            if isinstance(sc, dict):
                return sc.get(field, default)
            return getattr(sc, field, default)

        sc_idx = _get("index") or idx

        # Entity bindings extraction
        bindings = _get("entity_bindings")
        if hasattr(bindings, "model_dump"):
            bindings = bindings.model_dump()
        elif hasattr(bindings, "dict"):
            bindings = bindings.dict()
        elif not isinstance(bindings, dict):
            bindings = {}

        char_id = str(bindings.get("character_id") or "").strip() or None
        env_id = str(bindings.get("environment_id") or "").strip() or None
        concept_id = str(bindings.get("concept_id") or "").strip() or None
        obj_ids = [str(o).strip() for o in (bindings.get("object_ids") or []) if str(o).strip()]

        # Check A: Character ID existence & restrictions
        if char_id:
            character_occurrences += 1
            if char_id not in known_character_ids:
                warnings.append(
                    f"Scene {idx} references character_id '{char_id}' which does not exist in the Video Bible entity catalog."
                )
            if human_presence == "none" or not known_character_ids:
                warnings.append(
                    f"Scene {idx} binds character '{char_id}', but Creative Direction specified human_presence='none'."
                )

        # Check A2: Character state without permitted character
        char_state = _get("character_state")
        if char_state and (human_presence == "none" or not known_character_ids or not char_id):
            warnings.append(
                f"Scene {idx} specifies character_state, but no character is permitted or bound."
            )

        # Check B: Environment ID existence
        if env_id and env_id not in known_environment_ids:
            warnings.append(
                f"Scene {idx} references environment_id '{env_id}' which does not exist in the Video Bible entity catalog."
            )

        # Check C: Object IDs existence
        for oid in obj_ids:
            if oid not in known_object_ids:
                warnings.append(
                    f"Scene {idx} references object_id '{oid}' which does not exist in the Video Bible entity catalog."
                )

        # Check D: Concept ID existence
        if concept_id:
            if concept_id not in known_concept_ids:
                warnings.append(
                    f"Scene {idx} references concept_id '{concept_id}' which does not exist in the Video Bible entity catalog."
                )

        # Check E: Motivated Environment Transitions
        transition = str(_get("transition_from_previous") or "establishing").strip()
        if idx > 1 and env_id and last_environment_id and env_id != last_environment_id:
            valid_location_transitions = {
                "location_change",
                "spatial_cutaway",
                "explanatory_cutaway",
                "surreal_metaphor",
                "resolution_return",
            }
            if transition not in valid_location_transitions:
                warnings.append(
                    f"Scene {idx} switches environment from '{last_environment_id}' to '{env_id}' "
                    f"under transition '{transition}'. Environment shifts require motivated transitions: "
                    f"{', '.join(sorted(valid_location_transitions))}."
                )
        if env_id:
            last_environment_id = env_id

        # Check E2: Cross-Environment Spatial Contradiction
        # If an environment is bound, ensure remote persistent landmarks from other environments
        # are not described as local physical horizon/in-scene fixtures without multi-location framing.
        if env_id and env_id in known_environment_ids:
            scene_text = f"{_get('visual_description') or ''} {_get('focal_point') or ''} {_get('image_prompt') or ''}".lower()
            orig_scene_text = f"{_get('visual_description') or ''} {_get('focal_point') or ''} {_get('image_prompt') or ''}"

            # Legitimate multi-location framing requires explicit split, dual, inset, global/orbital, or in-scene map composition.
            # Do NOT automatically exempt compositions merely containing words like "cutaway", "composite", or "visual metaphor".
            raw_shot = str(_get("shot_type") or "").lower()
            raw_tech = str(_get("visual_technique") or "").lower()
            allowed_multi_location = (
                any(t in raw_shot for t in ("split", "dual", "inset", "pip", "split_screen", "side_by_side", "dual_view"))
                or any(t in raw_tech for t in ("split", "dual", "inset", "pip", "split_screen", "side_by_side", "dual_view"))
                or any(k in scene_text for k in (
                    "split-screen", "split screen", "side-by-side", "dual view", "inset",
                    "in-scene map", "wall map", "framed painting", "map of", "drawing of", "chart of",
                    "global view", "orbital view", "satellite view", "view from space", "planetary view",
                    "split comparison"
                ))
            )
            if not allowed_multi_location:
                for other_env in (entities.get("environments") or []):
                    other_id = str(other_env.get("id") if isinstance(other_env, dict) else getattr(other_env, "id", "")).strip()
                    if not other_id or other_id == env_id:
                        continue

                    # Extract original name and props to identify proper noun landmarks vs generic features
                    orig_other_name = str(other_env.get("name") if isinstance(other_env, dict) else getattr(other_env, "name", "")).strip()
                    raw_props = (other_env.get("persistent_props") if isinstance(other_env, dict) else getattr(other_env, "persistent_props", [])) or []
                    orig_other_props = [str(p).strip() for p in raw_props]

                    # Comprehensive generic landscape / object terms that must NOT be treated as remote landmarks
                    generic_landscape_terms = {
                        "room", "wall", "walls", "floor", "door", "window", "table", "chair", "light", "cloud", "clouds",
                        "sky", "skies", "field", "fields", "land", "lands", "tree", "trees", "grass", "water", "road", "street", "house",
                        "building", "roof", "horizon", "ambient", "daylight", "night", "view", "deck", "upper", "lower",
                        "open", "surface", "band", "narrow", "wide", "simple", "dirt", "soil", "stone", "stones", "stonewall",
                        "wood", "timber", "fence", "mud", "dust", "smoke", "haze", "mist", "hill", "hills", "ridge", "slope",
                        "rock", "rocks", "path", "crop", "crops", "wheat", "rye", "farm", "farms", "barn", "cart", "wagon",
                        "frost", "snow", "cold", "sun", "sunlight", "shadow", "shadows", "village", "town", "coast", "shore"
                    }

                    # 1. Identify confirmed proper noun landmarks (capitalized proper nouns, e.g. "Tambora", "Sumbawa", "Caldera")
                    common_title_words = {
                        "The", "A", "An", "Mount", "Mt", "Island", "Islands", "North", "South", "East", "West", "Upper", "Lower",
                        "Rural", "Farmland", "Europe", "America", "Interior", "Exterior", "Room", "Valley", "River", "Ocean",
                        "Sea", "Lake", "Mountain", "Volcano", "Volcanic", "Farm", "Field", "Barn", "House", "Storeroom",
                        "Peak", "Peaks", "Rim", "Vents", "Vent", "Slope", "Slopes", "Coastline", "Beach", "Water", "Sky", "Skies",
                        "Crater", "Smoking", "Cloud", "Clouds", "Haze", "Ash", "Cold", "Period", "Stone", "Dry", "Walls", "Wall",
                        "Damaged", "Wheat", "Fields", "Massive", "Indonesian", "Rolling", "Agricultural"
                    }
                    confirmed_proper_landmarks: set[str] = set()
                    for text_src in [orig_other_name] + orig_other_props:
                        for word in re.findall(r"\b[A-Z][a-z]{3,}\b", text_src):
                            if word not in common_title_words and word.lower() not in generic_landscape_terms:
                                confirmed_proper_landmarks.add(word)

                    contradiction_found = False
                    for prop_word in sorted(confirmed_proper_landmarks, key=len, reverse=True):
                        if re.search(r"\b" + re.escape(prop_word) + r"\b", orig_scene_text, flags=re.IGNORECASE):
                            warnings.append(
                                f"[Contradiction] Scene {sc_idx} is bound to environment '{env_id}', but its visual description or focal point "
                                f"places remote landmark '{prop_word}' from environment '{other_id}' directly into the local scene "
                                "without explicit multi-location framing (e.g. split-screen, in-scene map, or global view)."
                            )
                            contradiction_found = True
                            break

                    if contradiction_found:
                        break

                    # 2. Identify uncertain candidate tokens (non-proper tokens) -> emit [Advisory]
                    uncertain_tokens: set[str] = set()
                    env_entry = next((e for e in (entities.get("environments") or []) if (str(e.get("id") if isinstance(e, dict) else getattr(e, "id", "")) == env_id)), None)
                    this_name = str(env_entry.get("name") if isinstance(env_entry, dict) else getattr(env_entry, "name", "")).lower() if env_entry else ""
                    for prop_str in [orig_other_name.lower()] + [p.lower() for p in orig_other_props]:
                        for word in re.findall(r"\b[a-z]{5,}\b", prop_str):
                            if word not in generic_landscape_terms and word not in {w.lower() for w in confirmed_proper_landmarks} and word not in this_name:
                                uncertain_tokens.add(word)

                    for ut in uncertain_tokens:
                        if re.search(r"\b" + re.escape(ut) + r"\b", scene_text):
                            warnings.append(
                                f"[Advisory] Scene {sc_idx} mentions '{ut}', which may relate to remote environment '{other_id}'. "
                                "Ensure it is not depicted as a local physical fixture."
                            )
                            break

        # Check F: Camera Shot Variety & Anti-Repetition
        camera_data = _get("camera")
        shot_type = ""
        if isinstance(camera_data, dict):
            shot_type = str(camera_data.get("shot_type") or "").strip()
        elif hasattr(camera_data, "shot_type"):
            shot_type = str(getattr(camera_data, "shot_type") or "").strip()
        if not shot_type:
            shot_type = str(_get("shot_type") or "medium").strip()

        if shot_type:
            shot_types_used.add(shot_type.lower())
            if shot_type.lower() == (last_shot_type or "").lower():
                consecutive_shot_repeats += 1
                if consecutive_shot_repeats >= 3:
                    warnings.append(
                        f"Scenes {idx-2} to {idx} repeat the same camera shot type '{shot_type}' without framing variety."
                    )
            else:
                consecutive_shot_repeats = 1
                last_shot_type = shot_type

        # Check F2: Diagram Repetition
        tech = str(_get("visual_technique") or "").lower()
        is_diagram = shot_type.lower() == "diagram" or any(t in tech for t in ("diagram", "schematic"))
        if is_diagram:
            consecutive_diagrams += 1
            current_concept = str((_get("entity_bindings") or {}).get("concept_id") if isinstance(_get("entity_bindings"), dict) else getattr(_get("entity_bindings"), "concept_id", "") or "")
            current_focal = str(_get("focal_point") or "").lower()
            current_desc = str(_get("visual_description") or "").lower()

            # Check if consecutive diagrams explain distinct sequential process stages
            # (different concept IDs, or distinct focal points / stage descriptions)
            is_distinct_stage = False
            if prev_diagram_info:
                prev_c, prev_f, prev_d = prev_diagram_info
                if current_concept and prev_c and current_concept != prev_c:
                    is_distinct_stage = True
                elif current_focal and prev_f:
                    f_words_curr = set(re.findall(r"\b[a-z]{4,}\b", current_focal))
                    f_words_prev = set(re.findall(r"\b[a-z]{4,}\b", prev_f))
                    if len(f_words_curr - f_words_prev) >= 2 or len(f_words_prev - f_words_curr) >= 2:
                        is_distinct_stage = True
                elif current_desc and prev_d:
                    d_words_curr = set(re.findall(r"\b[a-z]{4,}\b", current_desc))
                    d_words_prev = set(re.findall(r"\b[a-z]{4,}\b", prev_d))
                    if len(d_words_curr - d_words_prev) >= 3:
                        is_distinct_stage = True

            if is_distinct_stage:
                distinct_diagram_stages += 1
            else:
                redundant_diagrams += 1

            prev_diagram_info = (current_concept, current_focal, current_desc)

            # Emit advisory warning only if consecutive diagrams lack progressive informational value
            if consecutive_diagrams >= 3 and redundant_diagrams >= 2:
                warnings.append(
                    f"[Advisory] Scenes {idx-consecutive_diagrams+1} to {idx} contain {consecutive_diagrams} consecutive diagrams. "
                    "Ensure each diagram introduces a distinct sequential stage or new informational layer rather than repeating static schematics."
                )
        else:
            consecutive_diagrams = 0
            distinct_diagram_stages = 0
            redundant_diagrams = 0
            prev_diagram_info = None

    # Check G: Character role adherence across video
    if human_presence in ("single", "ensemble") and known_character_ids and character_occurrences == 0:
        warnings.append(
            f"Creative Direction specified featured characters (human_presence='{human_presence}'), but no scenes bound a character."
        )

    # Check H: Global framing diversity
    if len(scenes) >= 5 and len(shot_types_used) < 3:
        warnings.append(
            f"Storyboard uses only {len(shot_types_used)} distinct shot types across {len(scenes)} scenes. "
            "Aim for at least 3-4 distinct framing perspectives (e.g. wide, medium, close-up, cutaway)."
        )

    # Check I: Opening and Resolution Intent Quality
    if scenes:
        def _get_item(item: Any, field: str) -> Any:
            return item.get(field) if isinstance(item, dict) else getattr(item, field, None)

        first_intent = str(_get_item(scenes[0], "visual_intent") or "").strip()
        if first_intent and first_intent not in ("establishing", "action_event", "environmental_context"):
            warnings.append(
                f"[Advisory] Scene 1 visual_intent is '{first_intent}'. Opening beats should establish setting, scale, or dramatic hook "
                "(recommended: 'establishing', 'action_event', or 'environmental_context')."
            )

        if len(scenes) >= 2:
            last_intent = str(_get_item(scenes[-1], "visual_intent") or "").strip()
            if last_intent and last_intent not in ("narrative_resolution", "environmental_context", "human_experience"):
                warnings.append(
                    f"[Advisory] Scene {len(scenes)} visual_intent is '{last_intent}'. Concluding beats should provide narrative resolution or lasting closure "
                    "(recommended: 'narrative_resolution')."
                )

    return warnings
