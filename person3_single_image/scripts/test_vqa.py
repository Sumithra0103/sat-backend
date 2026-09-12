"""CLI test script to verify Single-Image VQA Service."""

import sys
import json
from pathlib import Path
from PIL import Image, ImageDraw

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.preprocessing.image_loader import load_image
from app.services.vqa_service import VQAService
from scripts.test_image import create_sample_image_if_needed


def main():
    print("=" * 60)
    print("SatQuery AI — Test VQA Service")
    print("=" * 60)

    # 1. Image
    sample_file = repo_root / "outputs" / "sample_satellite.png"
    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else create_sample_image_if_needed(sample_file)

    # 2. Question
    question = sys.argv[2] if len(sys.argv) > 2 else "What structures or water bodies are present in this scene?"

    print(f"Loading image from: {target_path}")
    print(f"Question: '{question}'")

    image = load_image(target_path)
    service = VQAService()
    result = service.process_question(image, question)

    print("\n--- VQA Service Output ---")
    print(json.dumps(result, indent=2))

    # Save answer to outputs/answers/
    answers_dir = repo_root / "outputs" / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    out_file = answers_dir / "vqa_result.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[INFO] Saved answer to: {out_file}")


if __name__ == "__main__":
    main()
