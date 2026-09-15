"""CLI test script to verify Visual Grounding Service."""

import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.preprocessing.image_loader import load_image
from app.services.grounding_service import GroundingService
from scripts.test_image import create_sample_image_if_needed


def main():
    print("=" * 60)
    print("SatQuery AI — Test Visual Grounding Service")
    print("=" * 60)

    sample_file = repo_root / "outputs" / "sample_satellite.png"
    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else create_sample_image_if_needed(sample_file)
    query = sys.argv[2] if len(sys.argv) > 2 else "locate rectangular building structures"

    print(f"Loading image from: {target_path}")
    print(f"Grounding query: '{query}'")

    image = load_image(target_path)
    service = GroundingService()
    result = service.process_grounding(image, query)

    print("\n--- Visual Grounding Service Output ---")
    print(json.dumps(result, indent=2))
    print("\nNote: Mock model reports empty boxes (does NOT invent fake coordinates).")


if __name__ == "__main__":
    main()
