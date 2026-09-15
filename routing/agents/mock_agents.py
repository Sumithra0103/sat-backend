"""
Pre-configured mock remote sensing specialist agents for registry seeding.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from schemas.agent_schema import (
    AgentRegisterRequest,
    AgentCapability,
    InputModality,
    InputFormat,
    AgentStatus
)


def get_mock_agent_requests():
    return [
        AgentRegisterRequest(
            agent_id="rs-vqa-agent",
            name="Remote Sensing Visual Question Answering Agent",
            version="1.2.0",
            description="Specialist VQA model adapted for remote sensing using RSVQA and VRSBench benchmarks.",
            capabilities=[AgentCapability.SINGLE_IMAGE_VQA],
            input_modalities=[InputModality.OPTICAL, InputModality.SAR],
            supported_formats=[InputFormat.GEOTIFF, InputFormat.TIFF, InputFormat.PNG, InputFormat.JPEG],
            endpoint_url="http://localhost:8001/v1/vqa",
            parameters_schema={"temperature": 0.2, "max_tokens": 128},
            metadata={"training_dataset": "RSVQA+VRSBench", "backbone": "RemoteCLIP-ViT-L", "latency_ms": 65},
            status=AgentStatus.ACTIVE
        ),
        AgentRegisterRequest(
            agent_id="rs-grounding-agent",
            name="Remote Sensing Text-Guided Region Grounding Agent",
            version="1.1.0",
            description="Text-prompted spatial bounding box and polygon detector for geospatial entities.",
            capabilities=[AgentCapability.REGION_GROUNDING],
            input_modalities=[InputModality.OPTICAL, InputModality.SAR],
            supported_formats=[InputFormat.GEOTIFF, InputFormat.TIFF, InputFormat.PNG],
            endpoint_url="http://localhost:8002/v1/grounding",
            parameters_schema={"box_threshold": 0.35, "text_threshold": 0.25},
            metadata={"backbone": "Grounding-DINO-RS", "latency_ms": 85, "spatial_output": "bbox_and_polygon"},
            status=AgentStatus.ACTIVE
        ),
        AgentRegisterRequest(
            agent_id="rs-caption-agent",
            name="Remote Sensing Scene Description & Captioning Agent",
            version="1.0.0",
            description="Generates detailed multi-scale descriptions and land-cover breakdowns fine-tuned on BigEarthNet.",
            capabilities=[AgentCapability.SCENE_CAPTIONING],
            input_modalities=[InputModality.OPTICAL, InputModality.SAR],
            supported_formats=[InputFormat.GEOTIFF, InputFormat.TIFF, InputFormat.PNG, InputFormat.JPEG],
            endpoint_url="http://localhost:8004/v1/caption",
            parameters_schema={"detail_level": "comprehensive", "include_landcover_stats": True},
            metadata={"training_dataset": "BigEarthNet-MM", "backbone": "RS-BLIP2", "latency_ms": 70},
            status=AgentStatus.ACTIVE
        ),
        AgentRegisterRequest(
            agent_id="rs-change-agent",
            name="Bi-Temporal Change Detection & CDVQA Specialist Agent",
            version="1.5.0",
            description="Analyzes bi-temporal image pairs to detect land-cover changes and answer change-related queries.",
            capabilities=[AgentCapability.BITEMPORAL_CHANGE_DETECTION, AgentCapability.CHANGE_VQA],
            input_modalities=[InputModality.BITEMPORAL_OPTICAL, InputModality.BITEMPORAL_SAR],
            supported_formats=[InputFormat.GEOTIFF, InputFormat.TIFF],
            endpoint_url="http://localhost:8003/v1/change",
            parameters_schema={"change_threshold": 0.5, "generate_change_mask": True},
            metadata={"training_dataset": "SECOND (Semantic Change Detection) + CDVQA", "backbone": "Bi-Temporal Siamese ViT", "latency_ms": 110},
            status=AgentStatus.ACTIVE
        ),
        AgentRegisterRequest(
            agent_id="rs-crossmodal-agent",
            name="Cross-Modal Optical-SAR Joint Analysis Specialist",
            version="1.3.0",
            description="Jointly processes co-registered Optical and SAR imagery for all-weather feature extraction.",
            capabilities=[AgentCapability.CROSS_MODAL_FUSION],
            input_modalities=[InputModality.CROSS_MODAL_OPTICAL_SAR],
            supported_formats=[InputFormat.GEOTIFF, InputFormat.TIFF],
            endpoint_url="http://localhost:8005/v1/crossmodal_fusion",
            parameters_schema={"fusion_strategy": "late_attention_fusion", "output_masks": ["built_up", "water"]},
            metadata={"cloud_resilience": True, "backbone": "SAR-Optical Cross-Attention Network", "latency_ms": 125},
            status=AgentStatus.ACTIVE
        )
    ]
