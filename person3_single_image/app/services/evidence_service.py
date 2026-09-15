"""Visual evidence generation service for remote sensing analysis.

Generates an annotated visual evidence image:
- If bounding boxes are present: draws highlighted bounding boxes and labels directly on the scene.
- If no bounding boxes are present: renders an informative text panel showing the query and answer below the scene.
Saves all outputs under outputs/evidence/.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont


# Default project outputs directory
DEFAULT_EVIDENCE_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "evidence"


class EvidenceService:
    """Service to create visual evidence images from remote sensing inferences."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_EVIDENCE_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_font(self, size: int = 16) -> ImageFont.ImageFont:
        """Loads a TrueType font if available, falling back to PIL default font."""
        try:
            # Common Windows fonts
            for font_name in ["arial.ttf", "segoeui.ttf", "calibri.ttf"]:
                try:
                    return ImageFont.truetype(font_name, size=size)
                except IOError:
                    continue
            return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

    def _wrap_text(self, text: str, max_chars: int = 70) -> List[str]:
        """Simple word wrapping for panel rendering."""
        words = text.split()
        if not words:
            return [""]
        lines = []
        current_line = []
        current_len = 0
        for word in words:
            if current_len + len(word) + (1 if current_line else 0) <= max_chars:
                current_line.append(word)
                current_len += len(word) + (1 if len(current_line) > 1 else 0)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_len = len(word)
        if current_line:
            lines.append(" ".join(current_line))
        return lines

    def create_evidence(
        self,
        image: Image.Image,
        query: str,
        answer: str,
        boxes: Optional[List[List[int]]] = None,
        labels: Optional[List[str]] = None,
        filename_prefix: str = "evidence",
    ) -> Dict[str, Any]:
        """Creates and saves a visual evidence image.

        Args:
            image: Original remote sensing PIL Image.
            query: Question or grounding query text.
            answer: Generated answer or description text.
            boxes: Optional list of bounding boxes [[x1, y1, x2, y2], ...].
            labels: Optional list of labels corresponding to bounding boxes.
            filename_prefix: Prefix for saved file.

        Returns:
            Dictionary containing:
            {
                "evidence_path": str,
                "type": "bounding_boxes" | "qa_panel",
                "boxes_count": int,
                "saved_filename": str
            }
        """
        timestamp = int(time.time() * 1000)
        boxes = boxes or []
        labels = labels or []

        if boxes:
            # Case 1: Draw bounding boxes on copy of image
            evidence_img = image.copy()
            draw = ImageDraw.Draw(evidence_img)
            font = self._get_font(size=14)

            # Palette for distinct box outlines
            palette = ["#00FF66", "#FF3366", "#33CCFF", "#FFCC00", "#CC33FF"]

            for i, box in enumerate(boxes):
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    color = palette[i % len(palette)]
                    # Draw thicker rectangle
                    for offset in range(3):
                        draw.rectangle(
                            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                            outline=color,
                        )
                    # Draw label if available
                    label_text = labels[i] if i < len(labels) else f"Target {i+1}"
                    # Background tag for readability
                    draw.rectangle([x1, max(0, y1 - 18), x1 + len(label_text) * 8 + 6, max(0, y1)], fill=color)
                    draw.text((x1 + 3, max(0, y1 - 16)), label_text, fill="black", font=font)

            evidence_type = "bounding_boxes"
        else:
            # Case 2: Create composite image with Q&A bottom panel
            width, height = image.size
            panel_padding = 16
            line_height = 20

            font_title = self._get_font(size=15)
            font_body = self._get_font(size=13)

            q_lines = self._wrap_text(f"Query: {query}", max_chars=max(40, width // 10))
            a_lines = self._wrap_text(f"Result: {answer}", max_chars=max(40, width // 10))

            panel_height = panel_padding * 2 + (len(q_lines) + len(a_lines)) * line_height + 10

            composite = Image.new("RGB", (width, height + panel_height), color=(20, 24, 33))
            # Paste original image at top
            composite.paste(image, (0, 0))

            draw = ImageDraw.Draw(composite)

            # Draw accent separation bar
            draw.line([(0, height), (width, height)], fill=(0, 168, 255), width=3)

            # Render text lines
            curr_y = height + panel_padding
            for line in q_lines:
                draw.text((panel_padding, curr_y), line, fill=(255, 215, 0), font=font_title)
                curr_y += line_height

            curr_y += 4
            for line in a_lines:
                draw.text((panel_padding, curr_y), line, fill=(240, 240, 245), font=font_body)
                curr_y += line_height

            evidence_img = composite
            evidence_type = "qa_panel"

        # Save result to outputs/evidence/
        out_filename = f"{filename_prefix}_{timestamp}.png"
        out_path = self.output_dir / out_filename
        evidence_img.save(out_path, format="PNG")

        return {
            "evidence_path": str(out_path),
            "type": evidence_type,
            "boxes_count": len(boxes),
            "saved_filename": out_filename,
        }
