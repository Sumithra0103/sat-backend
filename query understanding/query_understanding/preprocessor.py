"""
Input Preprocessor for SatQuery AI.
Sanitizes query text, validates image inputs and formats (GeoTIFF/TIFF/PNG/JPEG),
and parses project/user context.
"""

import os
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from .models import (
    ImageMetadata,
    ImageFormat,
    Modality,
    SensorType,
    ProjectContext,
    UserRole,
    ExecutionTraceStep,
)


class InputPreprocessor:
    """
    Handles initial input validation, text normalization, and image metadata format verification.
    """

    ALLOWED_FORMAT_EXTENSIONS = {
        ".tif": ImageFormat.GEOTIFF,
        ".tiff": ImageFormat.GEOTIFF,
        ".gtif": ImageFormat.GEOTIFF,
        ".png": ImageFormat.PNG,
        ".jpg": ImageFormat.JPEG,
        ".jpeg": ImageFormat.JPEG,
    }

    def preprocess_query(self, raw_query: str) -> Tuple[str, str, ExecutionTraceStep]:
        """
        Normalizes raw query string.
        Returns: (original_query, normalized_cleaned_query, trace_step)
        """
        if not raw_query or not raw_query.strip():
            raise ValueError("User query string cannot be empty.")

        cleaned = raw_query.strip()
        # Remove redundant whitespace and normalize unicode quotes
        cleaned_normalized = re.sub(r'\s+', ' ', cleaned)
        cleaned_normalized = cleaned_normalized.replace('“', '"').replace('”', '"').replace("’", "'")

        trace = ExecutionTraceStep(
            step="InputPreprocessor",
            status="SUCCESS",
            details=f"Query text normalized. Length: {len(cleaned_normalized)} chars."
        )

        return raw_query.strip(), cleaned_normalized, trace

    def validate_and_parse_images(self, raw_images: List[Any]) -> Tuple[List[ImageMetadata], List[ExecutionTraceStep]]:
        """
        Parses and validates raw image metadata dictionaries or objects.
        Checks formats: GeoTIFF, TIFF, PNG, JPEG.
        """
        parsed_images: List[ImageMetadata] = []
        traces: List[ExecutionTraceStep] = []

        for idx, img_input in enumerate(raw_images):
            if isinstance(img_input, ImageMetadata):
                img_meta = img_input
            elif isinstance(img_input, dict):
                img_meta = self._parse_dict_to_metadata(img_input, idx)
            else:
                raise TypeError(f"Invalid image metadata format at index {idx}. Expected dict or ImageMetadata.")

            # Validate Format
            if img_meta.format == ImageFormat.UNKNOWN:
                traces.append(ExecutionTraceStep(
                    step="InputPreprocessor",
                    status="WARNING",
                    details=f"Image [{img_meta.image_id}] '{img_meta.file_name}' has unrecognized format. Supported: GeoTIFF, TIFF, PNG, JPEG."
                ))
            else:
                traces.append(ExecutionTraceStep(
                    step="InputPreprocessor",
                    status="SUCCESS",
                    details=f"Validated image [{img_meta.image_id}] Format: {img_meta.format.value}, Modality: {img_meta.modality.value}, Sensor: {img_meta.sensor_type.value}."
                ))

            parsed_images.append(img_meta)

        return parsed_images, traces

    def parse_project_context(self, context_dict: Optional[Dict[str, Any]]) -> ProjectContext:
        """
        Constructs ProjectContext from input parameters.
        """
        if not context_dict:
            return ProjectContext(
                project_id="default_project",
                user_id="anonymous_user",
                user_role=UserRole.EDITOR,
                can_execute_heavy_models=True,
                permissions=["read", "write", "execute"]
            )

        role_str = str(context_dict.get("user_role", "EDITOR")).upper()
        try:
            role = UserRole(role_str)
        except ValueError:
            role = UserRole.EDITOR

        return ProjectContext(
            project_id=context_dict.get("project_id", "default_project"),
            user_id=context_dict.get("user_id", "user_default"),
            user_role=role,
            can_execute_heavy_models=context_dict.get("can_execute_heavy_models", True),
            permissions=context_dict.get("permissions", ["read", "write", "execute"])
        )

    def _parse_dict_to_metadata(self, data: Dict[str, Any], index: int) -> ImageMetadata:
        file_name = data.get("file_name", data.get("file_path", f"image_{index+1}"))
        ext = os.path.splitext(file_name)[1].lower()
        
        # Format detection
        fmt = self.ALLOWED_FORMAT_EXTENSIONS.get(ext, ImageFormat.UNKNOWN)
        if "format" in data:
            fmt_str = str(data["format"]).upper()
            if "GEOTIFF" in fmt_str or "TIFF" in fmt_str:
                fmt = ImageFormat.GEOTIFF
            elif "PNG" in fmt_str:
                fmt = ImageFormat.PNG
            elif "JPEG" in fmt_str or "JPG" in fmt_str:
                fmt = ImageFormat.JPEG

        # Modality detection
        modality_raw = str(data.get("modality", "")).upper()
        modality = Modality.UNKNOWN
        if "SAR" in modality_raw or "RADAR" in modality_raw or "RISAT" in modality_raw or "SENTINEL-1" in modality_raw:
            modality = Modality.SAR
        elif "OPTICAL" in modality_raw or "CARTOSAT" in modality_raw or "SENTINEL-2" in modality_raw or "RGB" in modality_raw:
            modality = Modality.OPTICAL
        elif "MULTISPECTRAL" in modality_raw:
            modality = Modality.MULTISPECTRAL

        # Fallback modality from filename keywords
        if modality == Modality.UNKNOWN:
            fn_lower = file_name.lower()
            if any(k in fn_lower for k in ["sar", "risat", "sentinel1", "radar", "_vv_", "_vh_"]):
                modality = Modality.SAR
            elif any(k in fn_lower for k in ["optical", "cartosat", "sentinel2", "rgb", "msi"]):
                modality = Modality.OPTICAL

        # Sensor detection
        sensor_raw = str(data.get("sensor_type", data.get("sensor", ""))).upper()
        sensor = SensorType.UNKNOWN
        if "CARTOSAT" in sensor_raw:
            sensor = SensorType.CARTOSAT_2S
        elif "RISAT" in sensor_raw:
            sensor = SensorType.RISAT
        elif "SENTINEL-1" in sensor_raw or "SENTINEL1" in sensor_raw:
            sensor = SensorType.SENTINEL_1
        elif "SENTINEL-2" in sensor_raw or "SENTINEL2" in sensor_raw:
            sensor = SensorType.SENTINEL_2
        elif "BIGEARTHNET" in sensor_raw or "VRSBENCH" in sensor_raw or "RSVQA" in sensor_raw:
            sensor = SensorType.PUBLIC_BENCHMARK
        elif modality == Modality.SAR:
            sensor = SensorType.GENERIC_SAR
        elif modality in (Modality.OPTICAL, Modality.MULTISPECTRAL):
            sensor = SensorType.GENERIC_OPTICAL

        return ImageMetadata(
            image_id=data.get("image_id", f"img_{index+1}"),
            file_name=file_name,
            format=fmt,
            modality=modality,
            sensor_type=sensor,
            timestamp=data.get("timestamp"),
            spatial_bounds=data.get("spatial_bounds"),
            resolution_m=data.get("resolution_m"),
            is_co_registered=data.get("is_co_registered", False)
        )
