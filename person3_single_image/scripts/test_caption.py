"""CLI test script to verify Single-Image Captioning Service."""

import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.preprocessing.image_loader import load_image
from app.services.caption_service import CaptionService
from scripts.test_image import create_sample_image_if_needed


def main():
    print("=" * 60)
    print("SatQuery AI — Test Captioning Service")
    print("=" * 60)

    sample_file = repo_root / "outputs" / "sample_satellite.png"
    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else create_sample_image_if_needed(sample_file)

    print(f"Loading image from: {target_path}")

    image = load_image(target_path)
    service = CaptionService()
    result = service.process_caption(image)

    print("\n--- Caption Service Output ---")
    print(json.dumps(result, indent=2))

    # Save caption to outputs/captions/
    captions_dir = repo_root / "outputs" / "captions"
    captions_dir.mkdir(parents=True, exist_ok=True)
    out_file = captions_dir / "caption_result.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[INFO] Saved caption to: {out_file}")


if __name__ == "__main__":
    main()
