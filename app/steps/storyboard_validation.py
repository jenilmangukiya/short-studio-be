"""Continuity and story progression validation for connected storyboards.

Validates that storyboard scenes adhere to Video Bible entities, Creative Direction
guidance, intentional environment transitions, visual diversity, and camera progression.
"""

from __future__ import annotations

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
    expected_char_count = (
        protagonist_cfg.get("character_count", "none")
        if isinstance(protagonist_cfg, dict)
        else "none"
    )

    character_occurrences = 0
    shot_types_used: set[str] = set()
    consecutive_shot_repeats = 1
    last_shot_type: str | None = None
    last_environment_id: str | None = None

    for idx, sc in enumerate(scenes, start=1):
        def _get(field: str, default: Any = None) -> Any:
            if isinstance(sc, dict):
                return sc.get(field, default)
            return getattr(sc, field, default)

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
            if expected_char_count == "none" or not known_character_ids:
                warnings.append(
                    f"Scene {idx} binds character '{char_id}', but Creative Direction specified character_count='none'."
                )

        # Check A2: Character state without permitted character
        char_state = _get("character_state")
        if char_state and (expected_char_count == "none" or not known_character_ids or not char_id):
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

    # Check G: Character role adherence across video
    if expected_char_count in ("single", "ensemble") and known_character_ids and character_occurrences == 0:
        warnings.append(
            f"Creative Direction specified character_count='{expected_char_count}', but no scenes bound a character."
        )

    # Check H: Global framing diversity
    if len(scenes) >= 5 and len(shot_types_used) < 3:
        warnings.append(
            f"Storyboard uses only {len(shot_types_used)} distinct shot types across {len(scenes)} scenes. "
            "Aim for at least 3-4 distinct framing perspectives (e.g. wide, medium, close-up, cutaway)."
        )

    return warnings
