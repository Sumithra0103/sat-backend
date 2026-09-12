"""CLI test script to verify unified analyze_single_image pipeline and task auto-detection."""

import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.services.pipeline_service import analyze_single_image
from scripts.test_image import create_sample_image_if_needed


def main():
    print("=" * 60)
    print("SatQuery AI — Test Unified Single-Image Pipeline")
    print("=" * 60)

    sample_file = repo_root / "outputs" / "sample_satellite.png"
    target_path = create_sample_image_if_needed(sample_file)

    test_cases = [
        {"query": "Is there a river crossing the terrain?", "task": None, "label": "Auto-detect VQA"},
        {"query": "describe this image in full detail", "task": None, "label": "Auto-detect Captioning"},
        {"query": "locate the storage tanks and buildings", "task": None, "label": "Auto-detect Grounding"},
        {"query": "Explicit VQA call", "task": "vqa", "label": "Explicit Task: VQA"},
    ]

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n--- [Case {idx}] {tc['label']} ---")
        print(f"Query: '{tc['query']}' | Explicit Task: {tc['task']}")
        result = analyze_single_image(
            image_path=target_path,
            query=tc["query"],
            task=tc["task"],
            generate_evidence=True,
        )
        print("Pipeline Response:")
        print(json.dumps(result, indent=2))
        evidence_list = result.get("evidence", [])
        if evidence_list:
            print(f"Evidence saved: {evidence_list[0].get('evidence_path')}")

    print("\n[SUCCESS] Unified pipeline tested across all modes.")


if __name__ == "__main__":
    main()
