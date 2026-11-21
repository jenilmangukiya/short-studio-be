"""Deterministic Reference Selection Engine for Shorts Studio."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .schemas import (
    ReferenceMetadata,
    ReferenceSelectionResult,
    RejectedReference,
    SelectedReference,
    VisualStylePreset,
)

logger = logging.getLogger(__name__)

DEFAULT_REFERENCE_BUDGET: int = 2


class ReferenceSelectionEngine:
    """Service that deterministically selects, filters, and prioritizes reference assets for scene generation."""

    def __init__(self, default_budget: int = DEFAULT_REFERENCE_BUDGET):
        self.default_budget = default_budget

    def select_references(
        self,
        scene: dict[str, Any] | Any,
        preset: VisualStylePreset,
        available_references: list[ReferenceMetadata] | None = None,
        budget: int | None = None,
        check_file_exists: bool = False,
        storage_root: Path | None = None,
    ) -> ReferenceSelectionResult:
        """Select up to `budget` reference images deterministically according to scene entity bindings.

        Priority order:
        1. Primary character, object, or concept bound to scene.
        2. Secondary character, machine, or environment bound to scene.
        3. Global style benchmark from preset assets (only if capacity remains and not crowded out by entities).
        """
        active_budget = budget or preset.capabilities.max_reference_budget or self.default_budget
        active_style_id = preset.id
        active_style_version = preset.version

        selected: list[SelectedReference] = []
        rejected: list[RejectedReference] = []
        seen_paths: set[str] = set()

        # 1. Capability check: if preset does not support reference images, reject all immediately
        if not preset.capabilities.supports_reference_images:
            reason_msg = f"Visual style preset '{preset.id}' does not support reference-conditioned generation."
            for ref in (available_references or []):
                rejected.append(
                    RejectedReference(
                        entity_id=ref.entity_id,
                        asset_path=ref.asset_path,
                        role=ref.role,
                        reason="unsupported_by_preset",
                        details=reason_msg,
                    )
                )
            return ReferenceSelectionResult(
                selected_references=[],
                rejected_references=rejected,
                budget=active_budget,
                active_style_id=active_style_id,
                summary=f"No references selected: {reason_msg}",
            )

        # 2. Extract scene entity bindings
        bindings = {}
        raw_bindings = scene.get("entity_bindings") if isinstance(scene, dict) else getattr(scene, "entity_bindings", None)
        if hasattr(raw_bindings, "model_dump"):
            bindings = raw_bindings.model_dump()
        elif hasattr(raw_bindings, "dict"):
            bindings = raw_bindings.dict()
        elif isinstance(raw_bindings, dict):
            bindings = dict(raw_bindings)

        bound_char_id = str(bindings.get("character_id") or "").strip() or None
        raw_sec_chars = bindings.get("secondary_character_ids") or (bindings.get("character_ids", [])[1:] if bindings.get("character_ids") else [])
        bound_sec_char_ids = [str(c).strip() for c in raw_sec_chars if str(c).strip() and str(c).strip() != bound_char_id]

        bound_env_id = str(bindings.get("environment_id") or "").strip() or None
        bound_concept_id = str(bindings.get("concept_id") or "").strip() or None

        raw_objs = list(bindings.get("object_ids") or [])
        raw_machines = list(bindings.get("machine_ids") or [])
        bound_machine_ids = [str(m).strip() for m in raw_machines if str(m).strip()]
        bound_obj_ids = [str(o).strip() for o in raw_objs if str(o).strip()]

        # 3. Index available entity references by entity_id
        refs_by_entity: dict[str, list[ReferenceMetadata]] = {}
        for raw_ref in (available_references or []):
            ref = ReferenceMetadata.normalize(raw_ref)
            refs_by_entity.setdefault(ref.entity_id, []).append(ref)

        def _is_file_valid(path_str: str) -> bool:
            if not check_file_exists or not path_str:
                return True
            p = Path(path_str)
            if p.is_absolute():
                return p.is_file()
            if storage_root:
                return (storage_root / p).is_file()
            return p.is_file()

        # Helper to evaluate candidate
        def _evaluate_candidate(
            ref: ReferenceMetadata, priority_tier: int, reason_prefix: str
        ) -> bool:
            # Check deduplication
            norm_path = str(ref.asset_path).strip()
            if not norm_path:
                return False
            if norm_path in seen_paths:
                rejected.append(
                    RejectedReference(
                        entity_id=ref.entity_id,
                        asset_path=norm_path,
                        role=str(ref.role or "unannotated"),
                        reason="duplicate_path",
                        details=f"Asset path '{norm_path}' already selected in this request.",
                    )
                )
                return False

            # Check file presence
            if not _is_file_valid(norm_path):
                rejected.append(
                    RejectedReference(
                        entity_id=ref.entity_id,
                        asset_path=norm_path,
                        role=str(ref.role or "unannotated"),
                        reason="file_not_found",
                        details=f"Reference asset file does not exist on disk: {norm_path}",
                    )
                )
                return False

            # Check style compatibility using detailed diagnostic reason
            compat, reason_detail = ref.check_compatibility(active_style_id, active_style_version)
            if not compat:
                rejected.append(
                    RejectedReference(
                        entity_id=ref.entity_id,
                        asset_path=norm_path,
                        role=str(ref.role or "unannotated"),
                        reason="incompatible_style",
                        details=reason_detail,
                    )
                )
                return False

            # Check budget
            if len(selected) >= active_budget:
                rejected.append(
                    RejectedReference(
                        entity_id=ref.entity_id,
                        asset_path=norm_path,
                        role=str(ref.role or "unannotated"),
                        reason="budget_exceeded",
                        details=(
                            f"Reference budget limit ({active_budget}) reached by higher-priority entity references. "
                            f"Cannot include {reason_prefix} reference for '{ref.entity_id}'."
                        ),
                    )
                )
                return False

            # Passed all checks -> select
            seen_paths.add(norm_path)
            selected.append(
                SelectedReference(
                    entity_id=ref.entity_id,
                    role=ref.role or "semantic",
                    asset_path=norm_path,
                    priority_tier=priority_tier,
                    selection_reason=reason_prefix,
                )
            )
            return True

        # ======================================================================
        # Priority Tier 1: Primary Character, Primary Concept, or Primary Object
        # ======================================================================
        if bound_char_id and bound_char_id in refs_by_entity:
            for ref in refs_by_entity[bound_char_id]:
                if _evaluate_candidate(ref, priority_tier=1, reason_prefix="Primary character"):
                    break
        elif bound_concept_id and bound_concept_id in refs_by_entity:
            for ref in refs_by_entity[bound_concept_id]:
                if _evaluate_candidate(ref, priority_tier=1, reason_prefix="Primary concept diagram"):
                    break
        elif bound_obj_ids and bound_obj_ids[0] in refs_by_entity:
            for ref in refs_by_entity[bound_obj_ids[0]]:
                if _evaluate_candidate(ref, priority_tier=1, reason_prefix="Primary object prop"):
                    break

        # ======================================================================
        # Priority Tier 2: Secondary Character, Machine, Environment, Props, or Concept
        # ======================================================================
        # 1. Secondary characters (when multiple characters are bound to scene)
        for sec_char_id in bound_sec_char_ids:
            if sec_char_id in refs_by_entity:
                for ref in refs_by_entity[sec_char_id]:
                    if _evaluate_candidate(ref, priority_tier=2, reason_prefix="Secondary character"):
                        break
                    break

        # 2. Machines (machines bound to the scene)
        remaining_machines = [m for m in bound_machine_ids if not any(s.entity_id == m for s in selected)]
        for m_id in remaining_machines:
            if m_id in refs_by_entity:
                for ref in refs_by_entity[m_id]:
                    if _evaluate_candidate(ref, priority_tier=2, reason_prefix="Machine"):
                        break
                    break

        # 3. Environment setting
        if bound_env_id and bound_env_id in refs_by_entity:
            for ref in refs_by_entity[bound_env_id]:
                if _evaluate_candidate(ref, priority_tier=2, reason_prefix="Environment"):
                    break
                break

        # 4. Secondary objects / props
        remaining_objs = [o for o in bound_obj_ids if not any(s.entity_id == o for s in selected)]
        for obj_id in remaining_objs:
            if obj_id in refs_by_entity:
                for ref in refs_by_entity[obj_id]:
                    if _evaluate_candidate(ref, priority_tier=2, reason_prefix="Secondary object prop"):
                        break
                    break

        # 5. Concept (if character or machine was primary and concept also bound)
        if bound_concept_id and not any(s.entity_id == bound_concept_id for s in selected):
            if bound_concept_id in refs_by_entity:
                for ref in refs_by_entity[bound_concept_id]:
                    if _evaluate_candidate(ref, priority_tier=2, reason_prefix="Secondary concept"):
                        break
                    break

        # ======================================================================
        # Priority Tier 3: Global Style Benchmark (Only if capacity remains)
        # ======================================================================
        benchmark_path = preset.assets.style_benchmark
        if benchmark_path:
            norm_benchmark = str(benchmark_path).strip()
            style_ref = ReferenceMetadata(
                entity_id="style_benchmark",
                role="style",
                asset_path=norm_benchmark,
                visual_style_id=preset.id,
                style_version=preset.version,
                description="Global visual style benchmark defining medium, linework, and lighting",
            )
            # If 2 higher-priority entity references already occupied budget, style_ref will be rejected with budget_exceeded
            _evaluate_candidate(style_ref, priority_tier=3, reason_prefix="Global style benchmark")

        summary_msg = (
            f"Selected {len(selected)}/{active_budget} references for style '{active_style_id}': "
            f"{[s.entity_id for s in selected]}. ({len(rejected)} rejected)"
        )
        logger.debug(summary_msg)

        return ReferenceSelectionResult(
            selected_references=selected,
            rejected_references=rejected,
            budget=active_budget,
            active_style_id=active_style_id,
            summary=summary_msg,
        )


_default_engine = ReferenceSelectionEngine()


def select_scene_references(
    scene: dict[str, Any] | Any,
    preset: VisualStylePreset,
    available_references: list[ReferenceMetadata] | None = None,
    budget: int | None = None,
    check_file_exists: bool = False,
    storage_root: Path | None = None,
) -> ReferenceSelectionResult:
    """Convenience helper using the default ReferenceSelectionEngine."""
    return _default_engine.select_references(
        scene=scene,
        preset=preset,
        available_references=available_references,
        budget=budget,
        check_file_exists=check_file_exists,
        storage_root=storage_root,
    )
