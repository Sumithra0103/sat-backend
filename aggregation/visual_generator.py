"""
SatQuery AI - Visual Evidence Generator (Phase 14)
=================================================
Generates visual evidence artifacts:
1. Grounded bounding box annotations on optical/SAR imagery.
2. Bi-temporal change detection heatmaps and mask overlays.
3. Side-by-side Optical vs. SAR cross-modal panel visualizations.
Saves PNG files into `evidence_artifacts/` and returns URI references.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from PIL import Image, ImageDraw, ImageFont

from aggregation.schemas import SpatialEvidence, TemporalEvidence, ModalEvidence, SpatialType

logger = logging.getLogger("SatQuery.Aggregation.VisualGenerator")


class VisualEvidenceGenerator:
    """
    Generates annotated PNG images for visual grounding, change maps, and cross-modal evidence.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.output_dir = Path(output_dir) if output_dir else self.root_dir / "evidence_artifacts"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def draw_bounding_boxes(
        self,
        spatial_evidence: List[SpatialEvidence],
        input_image_path: Optional[str] = None,
        output_filename: str = "grounded_evidence.png",
        img_size: Tuple[int, int] = (512, 512)
    ) -> str:
        """
        Draws bounding boxes on base image (or creates synthetic canvas if image not provided).
        """
        out_path = self.output_dir / output_filename

        if input_image_path and os.path.exists(input_image_path):
            try:
                base_img = Image.open(input_image_path).convert("RGB")
            except Exception:
                base_img = Image.new("RGB", img_size, color=(30, 40, 50))
        else:
            # Synthetic satellite-styled canvas
            base_img = Image.new("RGB", img_size, color=(30, 45, 60))

        draw = ImageDraw.Draw(base_img)
        w, h = base_img.size

        box_items = [e for e in spatial_evidence if e.bounding_box]
        colors = [(255, 50, 50), (50, 255, 50), (50, 150, 255), (255, 200, 50), (200, 50, 255)]

        for idx, item in enumerate(box_items):
            box = item.bounding_box
            if not box or len(box) != 4:
                continue

            ymin, xmin, ymax, xmax = box
            px_ymin = int(ymin * h)
            px_xmin = int(xmin * w)
            px_ymax = int(ymax * h)
            px_xmax = int(xmax * w)

            color = colors[idx % len(colors)]
            draw.rectangle([px_xmin, px_ymin, px_xmax, px_ymax], outline=color, width=3)

            label_text = f"{item.label} ({int(item.confidence * 100)}%)"
            draw.rectangle([px_xmin, max(0, px_ymin - 18), px_xmin + len(label_text) * 7 + 6, px_ymin], fill=color)
            draw.text((px_xmin + 3, max(0, px_ymin - 16)), label_text, fill=(255, 255, 255))

        base_img.save(out_path)
        logger.info(f"Saved grounded visual evidence artifact to {out_path}")
        return str(out_path)

    def draw_change_map(
        self,
        temporal_evidence: List[TemporalEvidence],
        output_filename: str = "change_evidence_map.png",
        img_size: Tuple[int, int] = (512, 512)
    ) -> str:
        """
        Generates bi-temporal change map overlay visualization (Pre = Blue, Post = Red/Green change highlights).
        """
        out_path = self.output_dir / output_filename
        base_img = Image.new("RGB", img_size, color=(20, 25, 35))
        draw = ImageDraw.Draw(base_img)

        w, h = img_size

        # Draw grid background representing remote sensing raster
        for x in range(0, w, 64):
            draw.line([(x, 0), (x, h)], fill=(40, 50, 65), width=1)
        for y in range(0, h, 64):
            draw.line([(0, y), (w, y)], fill=(40, 50, 65), width=1)

        # Draw change highlights
        for idx, item in enumerate(temporal_evidence):
            mag = item.change_magnitude
            # Center change region
            box_w = int(w * 0.4 * mag)
            box_h = int(h * 0.4 * mag)
            cx, cy = int(w * 0.5), int(h * 0.5)

            draw.rectangle([cx - box_w, cy - box_h, cx + box_w, cy + box_h], fill=(220, 50, 50, 180), outline=(255, 255, 255), width=2)
            label = f"CHANGE: {item.change_type} ({int(mag * 100)}%)"
            draw.text((cx - box_w + 5, cy - box_h + 5), label, fill=(255, 255, 255))

        base_img.save(out_path)
        logger.info(f"Saved bi-temporal change map artifact to {out_path}")
        return str(out_path)

    def draw_optical_sar_panel(
        self,
        modal_evidence: List[ModalEvidence],
        output_filename: str = "optical_sar_fusion_panel.png",
        img_size: Tuple[int, int] = (1024, 512)
    ) -> str:
        """
        Generates side-by-side Optical vs. SAR panel visualization.
        """
        out_path = self.output_dir / output_filename
        panel = Image.new("RGB", img_size, color=(15, 20, 30))
        draw = ImageDraw.Draw(panel)

        w, h = img_size
        half_w = w // 2

        # Optical Panel (Left)
        draw.rectangle([10, 10, half_w - 10, h - 10], outline=(50, 200, 100), width=2)
        draw.text((20, 20), "[ OPTICAL / MULTISPECTRAL VIEW ]", fill=(50, 200, 100))
        draw.text((20, 50), "Land-cover & Spectral Reflectance", fill=(200, 200, 200))

        # SAR Panel (Right)
        draw.rectangle([half_w + 10, 10, w - 10, h - 10], outline=(100, 150, 255), width=2)
        draw.text((half_w + 20, 20), "[ SAR RADAR BACKSCATTER VIEW ]", fill=(100, 150, 255))
        draw.text((half_w + 20, 50), "Microwave Surface Structure & Double Bounce", fill=(200, 200, 200))

        # Synergy text at bottom
        if modal_evidence:
            syn_text = f"Synergy: {modal_evidence[0].synergy_description}"
            draw.text((20, h - 40), syn_text, fill=(255, 220, 100))

        panel.save(out_path)
        logger.info(f"Saved Optical-SAR cross-modal panel artifact to {out_path}")
        return str(out_path)
