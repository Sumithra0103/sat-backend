"""
Mock Remote Sensing Specialist Agent Data Definitions for SatQuery AI.
These specifications are used to seed the Agent Registry with domain-adapted
specialist agents fine-tuned on BigEarthNet, RSVQA, VRSBench, and CDVQA datasets.
"""

from typing import List, Dict, Any
from schemas.agent_schema import (
    AgentRegisterRequest,
    AgentCapability,
    InputModality,
    InputFormat,
    AgentStatus
)

MOCK_SPECIALIST_AGENTS: List[Dict[str, Any]] = [
    {
        "agent_id": "rs-vqa-agent",
        "name": "SatQuery RS-VQA Specialist",
        "version": "1.2.0",
        "description": "Domain-adapted remote sensing visual question answering agent fine-tuned on BigEarthNet and RSVQA benchmarks for optical and SAR single images.",
        "capabilities": [AgentCapability.SINGLE_IMAGE_VQA, AgentCapability.SCENE_CAPTIONING],
        "input_modalities": [InputModality.OPTICAL, InputModality.SAR],
        "supported_formats": [InputFormat.GEOTIFF, InputFormat.TIFF, InputFormat.PNG, InputFormat.JPEG],
        "endpoint_url": "http://localhost:8001/v1/vqa/execute",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "max_tokens": {"type": "integer", "default": 256, "minimum": 16, "maximum": 1024},
                "temperature": {"type": "number", "default": 0.2, "minimum": 0.0, "maximum": 1.0},
                "confidence_threshold": {"type": "number", "default": 0.7, "minimum": 0.0, "maximum": 1.0}
            },
            "required": ["confidence_threshold"]
        },
        "metadata": {
            "training_dataset": "BigEarthNet.txt, RSVQA",
            "backbone": "ResNet-50 + RemoteSensing-Transformer",
            "spatial_resolution_m": 10,
            "supported_bands": ["B02", "B03", "B04", "B08", "VV", "VH"]
        },
        "status": AgentStatus.ACTIVE
    },
    {
        "agent_id": "rs-grounding-agent",
        "name": "SatQuery Text-Guided Region Grounding Agent",
        "version": "2.0.1",
        "description": "Specialist agent for land-cover scene captioning and natural language object/water/building region grounding and bounding box segmentation.",
        "capabilities": [AgentCapability.REGION_GROUNDING, AgentCapability.SCENE_CAPTIONING],
        "input_modalities": [InputModality.OPTICAL, InputModality.SAR],
        "supported_formats": [InputFormat.GEOTIFF, InputFormat.TIFF, InputFormat.PNG, InputFormat.JPEG],
        "endpoint_url": "http://localhost:8002/v1/grounding/execute",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "target_categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["water", "built-up", "forest", "agriculture"]
                },
                "output_mask_format": {"type": "string", "enum": ["geojson", "bbox", "mask_rle"], "default": "geojson"},
                "iou_threshold": {"type": "number", "default": 0.5}
            }
        },
        "metadata": {
            "training_dataset": "VRSBench, BigEarthNet-MM",
            "backbone": "Swin-B-GroundingDINO-RS",
            "supported_classes": ["water_body", "urban_structure", "vegetation", "bare_soil", "infrastructure"]
        },
        "status": AgentStatus.ACTIVE
    },
    {
        "agent_id": "rs-change-agent",
        "name": "SatQuery Bi-Temporal Change Detection & Change-VQA Agent",
        "version": "1.5.0",
        "description": "Multitemporal change detection agent capable of spatial change mapping, change description, and change-based VQA on bi-temporal image pairs.",
        "capabilities": [AgentCapability.BITEMPORAL_CHANGE_DETECTION, AgentCapability.CHANGE_VQA],
        "input_modalities": [InputModality.BITEMPORAL_OPTICAL, InputModality.BITEMPORAL_SAR],
        "supported_formats": [InputFormat.GEOTIFF, InputFormat.TIFF, InputFormat.PNG],
        "endpoint_url": "http://localhost:8003/v1/change/execute",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "generate_spatial_change_map": {"type": "boolean", "default": True},
                "change_sensitivity": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
                "temporal_delta_days": {"type": "integer", "description": "Acquisition interval in days"}
            }
        },
        "metadata": {
            "training_dataset": "SECOND (Semantic Change Detection), CDVQA, S2Looking",
            "backbone": "BiTemporal-SiamUnet-VLM",
            "evaluation_benchmarks": ["SECOND", "CDVQA", "ISRO-SAC-Multitemporal"]
        },
        "status": AgentStatus.ACTIVE
    },
    {
        "agent_id": "rs-crossmodal-agent",
        "name": "SatQuery Optical-SAR Cross-Modal Joint Analysis Agent",
        "version": "1.1.0",
        "description": "Cross-modal fusion specialist combining co-registered optical/multispectral spectral bands and SAR structural radar backscatter for cloud-resilient analysis.",
        "capabilities": [AgentCapability.CROSS_MODAL_FUSION, AgentCapability.SINGLE_IMAGE_VQA],
        "input_modalities": [InputModality.CROSS_MODAL_OPTICAL_SAR],
        "supported_formats": [InputFormat.GEOTIFF, InputFormat.TIFF],
        "endpoint_url": "http://localhost:8004/v1/crossmodal/execute",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "fusion_method": {"type": "string", "enum": ["early_concat", "cross_attention", "feature_pyramid"], "default": "cross_attention"},
                "sar_polarization": {"type": "array", "items": {"type": "string"}, "default": ["VV", "VH"]},
                "optical_cloud_masking": {"type": "boolean", "default": True}
            }
        },
        "metadata": {
            "training_dataset": "BigEarthNet-MM, Cartosat-2S + RISAT SAR (ISRO/SAC Benchmark)",
            "backbone": "OpticalSAR-CrossAttn-Net",
            "cloud_resilience": True
        },
        "status": AgentStatus.ACTIVE
    },
    {
        "agent_id": "rs-caption-agent",
        "name": "SatQuery Remote Sensing Scene Description & Captioning Agent",
        "version": "1.0.0",
        "description": "Domain-adapted remote sensing scene captioning and land-cover description model fine-tuned on BigEarthNet-MM.",
        "capabilities": [AgentCapability.SCENE_CAPTIONING, AgentCapability.SINGLE_IMAGE_VQA],
        "input_modalities": [InputModality.OPTICAL, InputModality.SAR],
        "supported_formats": [InputFormat.GEOTIFF, InputFormat.TIFF, InputFormat.PNG, InputFormat.JPEG],
        "endpoint_url": "http://localhost:8005/v1/caption/execute",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "detail_level": {"type": "string", "enum": ["brief", "standard", "comprehensive"], "default": "comprehensive"},
                "include_landcover_stats": {"type": "boolean", "default": True},
                "confidence_threshold": {"type": "number", "default": 0.6}
            }
        },
        "metadata": {
            "training_dataset": "BigEarthNet-MM, BigEarthNet.txt",
            "backbone": "RS-BLIP2-RemoteCLIP",
            "spatial_resolution_m": 10
        },
        "status": AgentStatus.ACTIVE
    }
]


def get_mock_agent_requests() -> List[AgentRegisterRequest]:
    """Converts mock dictionary definitions into validated AgentRegisterRequest instances."""
    return [AgentRegisterRequest(**agent_dict) for agent_dict in MOCK_SPECIALIST_AGENTS]
