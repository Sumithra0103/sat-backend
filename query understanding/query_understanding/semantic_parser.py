"""
Semantic Parser for SatQuery AI.
Extracts Remote Sensing domain entities (water body, built-up, land cover),
spatial action intents (highlight, describe, change, fuse), and temporal specifications.
"""

import re
from typing import Tuple, List, Optional, Set

from .models import (
    ExtractedEntities,
    ExecutionTraceStep
)


class SemanticParser:
    """
    Parses intent, targets, spatial requests, and temporal indicators from queries.
    """

    # Domain Land-Cover & Entity Taxonomy
    RS_ENTITY_PATTERNS = {
        "water body": [r'\bwater body\b', r'\bwater bodies\b', r'\bwater-covered\b', r'\bwater\b', r'\blake\b', r'\briver\b'],
        "built-up area": [r'\bbuilt-up area\b', r'\bbuilt-up\b', r'\burban\b', r'\bbuildings\b', r'\bsettlement\b', r'\bstructures\b'],
        "land-cover": [r'\bland-cover\b', r'\bland cover\b', r'\bland use\b', r'\bterrain\b', r'\blandscape\b'],
        "vegetation": [r'\bvegetation\b', r'\bforest\b', r'\btrees\b', r'\bcrops\b', r'\bagricultural\b', r'\bcanopy\b'],
        "road network": [r'\broad\b', r'\broads\b', r'\bhighway\b', r'\btransportation\b'],
    }

    # Spatial Action Intent Patterns
    SPATIAL_ACTION_PATTERNS = {
        "describe_caption": [r'\bdescribe\b', r'\bcaption\b', r'\bsummarize\b', r'\bwhat is visible\b', r'\bobjects visible\b'],
        "highlight_ground": [r'\bhighlight\b', r'\bsegment\b', r'\blocate\b', r'\bground\b', r'\bfind\b', r'\boutline\b', r'\bwhere is\b'],
        "change_detect": [r'\bwhat changed\b', r'\bwhere did the change occur\b', r'\bchange between\b', r'\bdifference between\b'],
        "change_quantify": [r'\bincreased\b', r'\bdecreased\b', r'\bremained unchanged\b', r'\bquantify change\b', r'\barea trend\b'],
        "cross_modal_fuse": [r'\btogether\b', r'\bcombine\b', r'\bfusion\b', r'\boptical and sar\b', r'\bsar and optical\b', r'\bjointly\b']
    }

    # Temporal Specification Keywords
    TEMPORAL_PATTERNS = [
        r'\bbetween these two dates\b',
        r'\bbetween dates\b',
        r'\bover time\b',
        r'\bbefore and after\b',
        r'\bdate 1\b',
        r'\bdate 2\b',
        r'\b\d{4}-\d{2}-\d{2}\b'
    ]

    def parse(
        self,
        query: str,
        coreference_resolved: bool = False,
        inherited_from_turn: Optional[int] = None
    ) -> Tuple[ExtractedEntities, ExecutionTraceStep]:
        """
        Extracts entities, spatial actions, and temporal specs from effective query.
        """
        query_lower = query.lower()

        # 1. Extract Target Entities
        extracted_targets: Set[str] = set()
        for entity_canonical, patterns in self.RS_ENTITY_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, query_lower):
                    extracted_targets.add(entity_canonical)
                    break

        # 2. Extract Spatial Actions
        extracted_actions: Set[str] = set()
        for action_name, patterns in self.SPATIAL_ACTION_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, query_lower):
                    extracted_actions.add(action_name)
                    break

        # 3. Extract Temporal Specifications
        extracted_temporal: List[str] = []
        for pat in self.TEMPORAL_PATTERNS:
            matches = re.findall(pat, query_lower)
            extracted_temporal.extend(matches)

        entities = ExtractedEntities(
            target_classes=sorted(list(extracted_targets)),
            spatial_actions=sorted(list(extracted_actions)),
            temporal_specs=extracted_temporal,
            is_coreference_resolved=coreference_resolved,
            inherited_from_turn=inherited_from_turn
        )

        trace = ExecutionTraceStep(
            step="SemanticParser",
            status="SUCCESS",
            details=(
                f"Targets: {entities.target_classes or 'None'}, "
                f"Actions: {entities.spatial_actions or 'None'}, "
                f"Temporal: {entities.temporal_specs or 'None'}."
            )
        )

        return entities, trace
