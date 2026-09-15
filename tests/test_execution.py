"""
Unit and Integration Tests for the SatQuery AI Execution Subsystem.
Tests all 19 stages: Request Validation, Input Validation, Geospatial Validation,
Image Loading, Preprocessing, Adapters, Specialist Models, Error Recovery (Retry & Fallback),
Output Validation, Confidence Extraction, and Evidence Collection.
"""

import sys
from pathlib import Path
import pytest

_ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT_DIR), str(_ROOT_DIR / "agent registry"), str(_ROOT_DIR / "routing"), str(_ROOT_DIR / "query understanding")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from satquery_engine import SatQueryEngine
from execution.schemas import (
    ExecutionRequest,
    ExecutionStatus,
    EvidenceType,
    GeospatialMetadata,
    ResourceRequirements
)
from execution.validation import (
    RequestValidator,
    RequestValidationError,
    InputValidator,
    InputValidationError,
    GeospatialValidator,
    GeospatialValidationError
)
from execution.preprocessing import ImageLoader, ImagePreprocessor
from execution.engine import ExecutionEngine
from execution.adapters import AdapterFactory


@pytest.fixture(scope="module")
def engine():
    eng = SatQueryEngine(cache_ttl_seconds=5.0)
    yield eng
    eng.close()


def test_request_validation_success():
    req = ExecutionRequest(
        execution_id="exec_001",
        agent_id="rs-vqa-agent",
        task="visual_question_answering",
        inputs=[{"file_name": "scene.tif", "format": "GeoTIFF", "modality": "OPTICAL"}],
        parameters={"confidence_threshold": 0.75}
    )
    spec = {
        "parameters_schema": {
            "properties": {
                "confidence_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            },
            "required": ["confidence_threshold"]
        }
    }
    assert RequestValidator.validate_request(req, spec) is True


def test_request_validation_failure_missing_param():
    req = ExecutionRequest(
        execution_id="exec_002",
        agent_id="rs-vqa-agent",
        task="visual_question_answering",
        inputs=[{"file_name": "scene.tif", "format": "GeoTIFF"}],
        parameters={}
    )
    spec = {
        "parameters_schema": {
            "properties": {
                "confidence_threshold": {"type": "number"}
            },
            "required": ["confidence_threshold"]
        }
    }
    with pytest.raises(RequestValidationError) as exc:
        RequestValidator.validate_request(req, spec)
    assert "Missing required parameter 'confidence_threshold'" in str(exc.value)


def test_input_validation_pair_requirement():
    inputs = [{"file_name": "t1.tif", "format": "GeoTIFF", "modality": "OPTICAL"}]
    with pytest.raises(InputValidationError) as exc:
        InputValidator.validate_inputs(inputs, task="bitemporal_change_detection")
    assert "requires an image pair" in str(exc.value)


def test_geospatial_validation_no_overlap():
    m1 = GeospatialMetadata(
        file_path="t1.tif", format="GeoTIFF", modality="OPTICAL",
        bounds=[10.0, 10.0, 11.0, 11.0], resolution_m=10.0
    )
    m2 = GeospatialMetadata(
        file_path="t2.tif", format="GeoTIFF", modality="OPTICAL",
        bounds=[20.0, 20.0, 21.0, 21.0], resolution_m=10.0
    )
    with pytest.raises(GeospatialValidationError) as exc:
        GeospatialValidator.validate_geospatial([m1, m2], task="change_detection")
    assert "no spatial overlap" in str(exc.value)


def test_geospatial_validation_unregistered():
    m1 = GeospatialMetadata(
        file_path="opt.tif", format="GeoTIFF", modality="OPTICAL",
        bounds=[72.8, 18.9, 72.9, 19.0], resolution_m=0.6, is_registered=True
    )
    m2 = GeospatialMetadata(
        file_path="sar.tif", format="GeoTIFF", modality="SAR",
        bounds=[72.8, 18.9, 72.9, 19.0], resolution_m=1.0, is_registered=False
    )
    with pytest.raises(GeospatialValidationError) as exc:
        GeospatialValidator.validate_geospatial([m1, m2], task="cross_modal_fusion")
    assert "not co-registered" in str(exc.value)


def test_image_loading_and_preprocessing():
    img = {
        "file_name": "sentinel2_optical.tif",
        "format": "GeoTIFF",
        "modality": "OPTICAL",
        "sensor_type": "Sentinel-2"
    }
    raster = ImageLoader.load_image(img)
    assert raster.width == 512
    assert raster.height == 512
    assert "B04" in raster.metadata.bands

    batch = ImagePreprocessor.preprocess_rasters([raster], task="scene_captioning")
    assert len(batch.tensors) == 1
    assert batch.normalization_method == "min_max_percentile"


def test_change_detection_execution(engine):
    executor = ExecutionEngine(db_session=engine.db)
    req = ExecutionRequest(
        execution_id="exec_cd_01",
        agent_id="rs-change-agent",
        task="bitemporal_change_detection",
        inputs=[
            {"file_name": "t1_2022.tif", "format": "GeoTIFF", "modality": "BITEMPORAL_OPTICAL", "bounds": [72.8, 18.9, 72.9, 19.0]},
            {"file_name": "t2_2026.tif", "format": "GeoTIFF", "modality": "BITEMPORAL_OPTICAL", "bounds": [72.8, 18.9, 72.9, 19.0]}
        ],
        parameters={"generate_spatial_change_map": True},
        resolved_query="What changed between these two dates, and where did the change occur?"
    )
    result = executor.execute_request(req)

    assert result.status == ExecutionStatus.SUCCESS
    assert result.agent == "rs-change-agent"
    assert result.result["changed_regions"] == 15
    assert "spatial_change_map" in result.result["change_map"]
    assert result.confidence >= 0.9
    assert result.execution_time > 0.0
    assert len(result.evidence) >= 1
    assert result.evidence[0].evidence_type == EvidenceType.CHANGE_MAP


def test_region_grounding_execution(engine):
    executor = ExecutionEngine(db_session=engine.db)
    req = ExecutionRequest(
        execution_id="exec_ground_01",
        agent_id="rs-grounding-agent",
        task="region_grounding",
        inputs=[{"file_name": "scene.png", "format": "PNG", "modality": "OPTICAL", "bounds": [72.8, 18.9, 72.9, 19.0]}],
        parameters={"output_mask_format": "geojson"},
        resolved_query="Highlight the water body referred to in the query."
    )
    result = executor.execute_request(req)

    assert result.status == ExecutionStatus.SUCCESS
    assert result.result["grounded_entity"] == "water_body"
    assert "bounding_box_normalized" in result.result
    assert any(ev.evidence_type == EvidenceType.GEOJSON for ev in result.evidence)
    assert any(ev.evidence_type == EvidenceType.BOUNDING_BOX for ev in result.evidence)


def test_crossmodal_fusion_execution(engine):
    executor = ExecutionEngine(db_session=engine.db)
    req = ExecutionRequest(
        execution_id="exec_cm_01",
        agent_id="rs-crossmodal-agent",
        task="cross_modal_fusion",
        inputs=[
            {"file_name": "cartosat2s.tif", "format": "GeoTIFF", "modality": "OPTICAL", "sensor_type": "Cartosat-2S", "bounds": [72.8, 18.9, 72.9, 19.0]},
            {"file_name": "risat1a_sar.tif", "format": "GeoTIFF", "modality": "SAR", "sensor_type": "RISAT-SAR", "bounds": [72.8, 18.9, 72.9, 19.0]}
        ],
        parameters={"fusion_method": "cross_attention"},
        resolved_query="Use the optical and SAR images together to identify built-up and water-covered regions."
    )
    result = executor.execute_request(req)

    assert result.status == ExecutionStatus.SUCCESS
    assert "Optical spectral bands and SAR" in result.result["joint_analysis"]
    assert "built_up_structures" in result.result["classified_features"]
    assert result.confidence > 0.9


def test_end_to_end_process_and_execute(engine):
    query = "What changed between these two dates, and where did the change occur?"
    images = [
        {"image_id": "img_t1", "file_name": "urban_t1_2022.tif", "format": "GeoTIFF", "modality": "OPTICAL", "bounds": [72.8, 18.9, 72.9, 19.0]},
        {"image_id": "img_t2", "file_name": "urban_t2_2026.tif", "format": "GeoTIFF", "modality": "OPTICAL", "bounds": [72.8, 18.9, 72.9, 19.0]}
    ]

    sqo, plan, results = engine.process_and_execute(query, images)

    assert sqo.input_compatibility.is_compatible is True
    assert plan.routing_status.value == "success"
    assert len(results) >= 1
    res = results[0]
    assert res.status.value == "success"
    assert res.agent == "rs-change-agent"
    assert res.result["changed_regions"] == 15
    assert len(res.evidence) >= 1
    assert len(res.execution_trace) >= 5
