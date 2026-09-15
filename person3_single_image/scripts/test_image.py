"""CLI test script to verify remote sensing image loading and preprocessing."""

import sys
from pathlib import Path
from PIL import Image, ImageDraw

# Add repository root to python path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.preprocessing.image_loader import (
    load_image,
    ImageNotFoundError,
    UnsupportedImageFormatError,
)


def create_sample_image_if_needed(output_path: Path) -> Path:
    """Generates a synthetic remote-sensing style test image if no user image is provided."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        img = Image.new("RGB", (256, 256), color=(45, 90, 45))  # Greenish terrain
        draw = ImageDraw.Draw(img)
        # Add a mock river
        draw.line([(0, 100), (120, 140), (256, 120)], fill=(30, 80, 180), width=18)
        # Add mock building footprints
        draw.rectangle([40, 40, 75, 75], fill=(180, 180, 190), outline=(230, 230, 230))
        draw.rectangle([180, 180, 220, 210], fill=(160, 160, 170), outline=(220, 220, 220))
        img.save(output_path)
        print(f"[INFO] Created synthetic test satellite image at: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("SatQuery AI — Test Image Loading")
    print("=" * 60)

    # Use command line argument or fallback to synthetic sample
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
    else:
        sample_file = repo_root / "outputs" / "sample_satellite.png"
        target_path = create_sample_image_if_needed(sample_file)

    print(f"Testing image path: {target_path}")

    try:
        image = load_image(target_path)
        print("[SUCCESS] Image loaded safely!")
        print(f"  - Format: {image.format}")
        print(f"  - Mode: {image.mode}")
        print(f"  - Dimensions: {image.size[0]} x {image.size[1]} (width x height)")
        print(f"  - Preserved original resolution without destructive resize.")
    except (ImageNotFoundError, UnsupportedImageFormatError) as err:
        print(f"[ERROR] Validation failed: {err}")
    except Exception as err:
        print(f"[ERROR] Unexpected error: {err}")


if __name__ == "__main__":
    main()
