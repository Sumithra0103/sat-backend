"""CLI script to test or verify RS-LLaVA configuration safely.

SAFETY NOTE:
This script WILL NOT download or load the 7B model unless the '--run-real' flag
is explicitly provided. Running on a GTX 1650 (4GB VRAM) without adequate GPU resources
is strongly discouraged for 7B inference.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.config import get_settings
from app.preprocessing.image_loader import load_image
from scripts.test_image import create_sample_image_if_needed


def main():
    parser = argparse.ArgumentParser(
        description="Safe test & verification tool for RS-LLaVA integration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=str, default=None, help="Path to remote sensing image.")
    parser.add_argument(
        "--question",
        type=str,
        default="What structures or terrain features are present in this scene?",
        help="Question for VQA.",
    )
    parser.add_argument(
        "--run-real",
        action="store_true",
        help="EXPLICIT FLAG REQUIRED to trigger actual 7B model download/inference.",
    )

    args = parser.parse_args()
    settings = get_settings()

    print("=" * 70)
    print("SatQuery AI — RS-LLaVA Integration Checker & Tester")
    print("=" * 70)
    print(f"Configured LoRA Model: {settings.model_name}")
    print(f"Configured Base Model: {settings.model_base}")
    print(f"Device:                {settings.device}")
    print(f"Torch DType:           {settings.torch_dtype}")
    print(f"Load in 4-bit:         {settings.load_in_4bit}")
    print(f"Load in 8-bit:         {settings.load_in_8bit}")
    print(f"Real Model Enabled:    {settings.use_real_rsllava} (SATQUERY_USE_REAL_RSLLAVA)")
    print("-" * 70)

    # Check dependency presence without downloading weights
    try:
        import torch
        import transformers
        deps_available = True
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        deps_available = False
        torch_version = "Not installed"
        cuda_available = False

    print(f"PyTorch installed:     {deps_available} ({torch_version})")
    print(f"CUDA Available:        {cuda_available}")
    if cuda_available:
        print(f"CUDA Device Name:      {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available VRAM:        {vram_gb:.2f} GB")
    print("=" * 70)

    # Guard: Require explicit --run-real flag
    if not args.run_real:
        print("\n[SAFETY GUARD ACTIVE] The 7B model will NOT be downloaded or loaded.")
        print("Real model inference requires:")
        print("  1. A suitable GPU machine (e.g. >=16GB VRAM for 16-bit, >=8GB VRAM for 4-bit)")
        print("  2. Enabling real-model mode: $env:SATQUERY_USE_REAL_RSLLAVA=\"true\" (Windows)")
        print("                              export SATQUERY_USE_REAL_RSLLAVA=true  (Linux/Mac)")
        print("  3. Installing optional packages: pip install -r requirements-rsllava.txt")
        print("  4. Running with the explicit flag:")
        print("       python scripts/test_rsllava.py --run-real\n")
        print("Hardware Notice:")
        print("  - RS-LLaVA-v1.5-7b is a 7-Billion parameter multimodal neural network.")
        print("  - Running on a standard 4GB VRAM laptop (GTX 1650) is NOT recommended.")
        print("  - Standard mock mode (SATQUERY_USE_REAL_RSLLAVA=false) allows complete development,")
        print("    API validation, and frontend integration on any hardware.\n")
        sys.exit(0)

    # If --run-real is explicitly passed:
    print("\n[WARNING] Explicit real inference requested (--run-real).")
    if not deps_available:
        print("[ERROR] Real model dependencies missing. Install requirements-rsllava.txt first:")
        print("  pip install -r requirements-rsllava.txt")
        sys.exit(1)

    sample_file = repo_root / "outputs" / "sample_satellite.png"
    target_path = Path(args.image) if args.image else create_sample_image_if_needed(sample_file)
    print(f"Loading image from: {target_path}")
    image = load_image(target_path)

    from app.models.vqa_model import RSLLaVAVQA
    from app.models.caption_model import RSLLaVACaption

    print("Initializing RSLLaVAVQA adapter (loading model weights)...")
    vqa_adapter = RSLLaVAVQA()
    print(f"Executing real VQA query: '{args.question}'...")
    vqa_res = vqa_adapter.answer(image, args.question)
    print("\n--- Real RS-LLaVA VQA Result ---")
    print(f"Answer: {vqa_res.get('answer')}")

    print("\nExecuting real caption generation...")
    cap_adapter = RSLLaVACaption()
    cap_res = cap_adapter.generate(image)
    print("\n--- Real RS-LLaVA Caption Result ---")
    print(f"Caption: {cap_res.get('caption')}")


if __name__ == "__main__":
    main()
