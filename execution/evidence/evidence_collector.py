"""
Evidence Collector for SatQuery AI Execution Subsystem.
Extracts, structures, and persists visual evidence:
- Spatial change maps (.png / .tif)
- Text-guided bounding boxes & segmentation masks
- GeoJSON spatial feature representations
- Confidence heatmaps and data source references
"""

import uuid
from typing import List, Dict, Any, Optional
from execution.schemas import EvidenceItem, EvidenceType, ExecutionTraceItem


class EvidenceCollector:
    """
    Collects and packages spatial, visual, and textual evidence
    from specialist model predictions.
    """

    @classmethod
    def collect_evidence(
        cls,
        agent_id: str,
        task: str,
        prediction: Dict[str, Any],
        image_inputs: List[Dict[str, Any]],
        trace: Optional[List[ExecutionTraceItem]] = None
    ) -> List[EvidenceItem]:
        items: List[EvidenceItem] = []

        source_ref = ", ".join([img.get("file_name", "image") for img in image_inputs])

        # 1. Change map evidence
        if "change_map" in prediction:
            items.append(EvidenceItem(
                evidence_id=f"ev_cmap_{uuid.uuid4().hex[:8]}",
                evidence_type=EvidenceType.CHANGE_MAP,
                label="Bi-Temporal Spatial Change Map",
                description=f"Identified {prediction.get('changed_regions', 0)} change clusters over {prediction.get('changed_area_km2', 0)} km².",
                confidence=float(prediction.get("confidence", 0.9)),
                file_path=prediction.get("change_map"),
                raw_metrics={
                    "changed_regions": prediction.get("changed_regions"),
                    "changed_area_km2": prediction.get("changed_area_km2"),
                    "change_trend": prediction.get("change_trend")
                },
                source_reference=source_ref
            ))

        # 2. Grounding / Bounding Box evidence
        if "bounding_box_normalized" in prediction:
            items.append(EvidenceItem(
                evidence_id=f"ev_bbox_{uuid.uuid4().hex[:8]}",
                evidence_type=EvidenceType.BOUNDING_BOX,
                label=prediction.get("label", "Grounded Region"),
                description=f"Bounding box coordinates for entity '{prediction.get('grounded_entity')}'.",
                confidence=float(prediction.get("iou_confidence", 0.88)),
                bounding_box=prediction.get("bounding_box_normalized"),
                source_reference=source_ref
            ))

        # 3. GeoJSON polygon evidence
        if "geojson_feature" in prediction:
            items.append(EvidenceItem(
                evidence_id=f"ev_geojson_{uuid.uuid4().hex[:8]}",
                evidence_type=EvidenceType.GEOJSON,
                label="Geospatial Polygon Geometry",
                description="Vector polygon boundary compatible with GIS tools (QGIS, ArcGIS, Leaflet).",
                confidence=float(prediction.get("iou_confidence", 0.88)),
                geojson=prediction.get("geojson_feature"),
                source_reference=source_ref
            ))

        # 4. Cross-Modal Fusion evidence
        if "identified_classes" in prediction:
            items.append(EvidenceItem(
                evidence_id=f"ev_crossmodal_{uuid.uuid4().hex[:8]}",
                evidence_type=EvidenceType.CONFIDENCE_HEATMAP,
                label="Optical-SAR Joint Feature Layer",
                description="Cross-modal attention map isolating cloud-penetrated urban and hydrological structures.",
                confidence=float(prediction.get("confidence", 0.94)),
                raw_metrics=prediction.get("identified_classes", {}),
                source_reference=source_ref
            ))

        if trace is not None:
            trace.append(ExecutionTraceItem(
                stage="Evidence Collection",
                action="aggregate_evidence_items",
                status="success",
                details={
                    "evidence_count": len(items),
                    "types": [e.evidence_type.value for e in items]
                }
            ))

        return items
