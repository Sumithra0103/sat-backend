"""
Context Resolver for SatQuery AI.
Manages multi-turn conversation session history, resolves image context inheritance,
analyzes spatial/temporal/modality pairing, and performs coreference resolution across queries.
"""

from typing import List, Dict, Any, Tuple, Optional
import re

from .models import (
    ImageMetadata,
    Modality,
    ConversationSession,
    ConversationTurn,
    StructuredQueryObject,
    ExecutionTraceStep,
    ProjectContext
)


class ContextResolver:
    """
    Resolves session history, image pairing (Bi-temporal, Cross-modal Optical+SAR),
    and coreference resolution ('it', 'this region', 'that change').
    """

    PRONOUN_PATTERNS = [
        r'\bit\b',
        r'\bthis region\b',
        r'\bthis area\b',
        r'\bthat region\b',
        r'\bthat area\b',
        r'\bthe same region\b',
        r'\bthe water body referred to\b',
        r'\bthat feature\b'
    ]

    def resolve_context(
        self,
        cleaned_query: str,
        current_images: List[ImageMetadata],
        session: ConversationSession,
        project_context: ProjectContext
    ) -> Tuple[List[ImageMetadata], str, bool, Optional[int], ExecutionTraceStep]:
        """
        Main context resolution entry point.
        Returns:
            effective_images: List[ImageMetadata] (current or inherited from session)
            effective_query: str (query enriched with resolved coreferences)
            coreference_resolved: bool
            inherited_from_turn: Optional[int]
            trace_step: ExecutionTraceStep
        """
        inherited_turn_idx: Optional[int] = None
        images_to_use = list(current_images)

        # 1. Image Inheritance Check
        if not images_to_use and session.active_images:
            images_to_use = list(session.active_images)
            inherited_turn_idx = session.turns[-1].turn_index if session.turns else 0

        # Update active images in session
        if current_images:
            session.active_images = list(current_images)

        # 2. Coreference & Entity Inheritance Resolution
        effective_query = cleaned_query
        coreference_found = False

        if session.turns:
            last_sqo: StructuredQueryObject = session.turns[-1].sqo
            prior_targets = last_sqo.extracted_entities.target_classes

            # Check if current query uses pronouns or vague references
            contains_pronoun = any(re.search(pat, cleaned_query, re.IGNORECASE) for pat in self.PRONOUN_PATTERNS)
            
            # Or if query is a short follow-up (e.g. "Has it increased?")
            if (contains_pronoun or "increased" in cleaned_query.lower() or "decreased" in cleaned_query.lower()) and prior_targets:
                primary_target = prior_targets[0]
                # Replace 'it' or inject primary target
                if re.search(r'\bit\b', cleaned_query, re.IGNORECASE):
                    effective_query = re.sub(r'\bit\b', primary_target, cleaned_query, flags=re.IGNORECASE)
                    coreference_found = True
                elif not any(target in cleaned_query.lower() for target in prior_targets):
                    effective_query = f"{cleaned_query} (referring to target: '{primary_target}')"
                    coreference_found = True

                if inherited_turn_idx is None:
                    inherited_turn_idx = session.turns[-1].turn_index

        trace_details = (
            f"Active Images: {len(images_to_use)} "
            f"({'Inherited from turn ' + str(inherited_turn_idx) if inherited_turn_idx is not None else 'Directly attached'}). "
            f"Coreference resolved: {coreference_found}."
        )

        trace = ExecutionTraceStep(
            step="ContextResolver",
            status="SUCCESS",
            details=trace_details
        )

        return images_to_use, effective_query, coreference_found, inherited_turn_idx, trace

    def determine_temporal_and_modal_structure(
        self, images: List[ImageMetadata]
    ) -> Tuple[str, List[str], List[str], List[str]]:
        """
        Analyzes modalities and image timestamps.
        Returns:
            temporal_structure: "SINGLE_TEMPORAL" | "BI_TEMPORAL" | "MULTI_TEMPORAL" | "CROSS_MODAL"
            detected_modalities: List[str]
            formats: List[str]
            sensors: List[str]
        """
        if not images:
            return "SINGLE_TEMPORAL", [], [], []

        modalities = [img.modality.value for img in images]
        formats = [img.format.value for img in images]
        sensors = [img.sensor_type.value for img in images]

        timestamps = [img.timestamp for img in images if img.timestamp]
        unique_timestamps = set(timestamps)

        has_optical = any(img.modality in (Modality.OPTICAL, Modality.MULTISPECTRAL) for img in images)
        has_sar = any(img.modality == Modality.SAR for img in images)

        # Check cross-modal
        if has_optical and has_sar and len(images) >= 2:
            temporal_structure = "CROSS_MODAL"
        elif len(unique_timestamps) >= 2 or len(images) == 2:
            temporal_structure = "BI_TEMPORAL"
        elif len(images) > 2:
            temporal_structure = "MULTI_TEMPORAL"
        else:
            temporal_structure = "SINGLE_TEMPORAL"

        return temporal_structure, modalities, formats, sensors
