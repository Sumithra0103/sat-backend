from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    DEVICE = "cpu"

if HAS_TORCH:
    try:
        from model.change_model import SiameseResNet18ChangeDetector
    except ImportError:
        from change_analysis.model.change_model import SiameseResNet18ChangeDetector
else:
    SiameseResNet18ChangeDetector = None


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT
    / "outputs"
    / "change_detection_v2"
    / "best_second_v2.pth"
)


class V2ChangeDetector:

    def __init__(
        self,
        checkpoint=CHECKPOINT,
        threshold=0.5,
    ):

        self.threshold = threshold
        self.has_torch = HAS_TORCH

        if self.has_torch and SiameseResNet18ChangeDetector is not None:
            try:
                self.model = (
                    SiameseResNet18ChangeDetector()
                    .to(DEVICE)
                )

                checkpoint_path = Path(checkpoint) if checkpoint else None
                if checkpoint_path and checkpoint_path.exists():
                    checkpoint_data = torch.load(
                        checkpoint_path,
                        map_location=DEVICE,
                        weights_only=False
                    )

                    self.model.load_state_dict(
                        checkpoint_data.get("model_state_dict", checkpoint_data)
                    )

                self.model.eval()
            except Exception:
                self.model = None
        else:
            self.model = None

    def load_image(self, path):

        if isinstance(path, (str, Path)):
            image = Image.open(path).convert("RGB")
        elif isinstance(path, Image.Image):
            image = path.convert("RGB")
        elif isinstance(path, np.ndarray):
            arr = path
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            image = Image.fromarray(arr).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(path)}")

        original_size = (
            image.height,
            image.width
        )

        image_resized = image.resize(
            (256, 256),
            Image.Resampling.BILINEAR
        )

        array = (
            np.asarray(
                image_resized,
                dtype=np.float32
            )
            / 255.0
        )

        if self.has_torch and self.model is not None:
            tensor = (
                torch.from_numpy(
                    array
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(DEVICE)
            )
            return tensor, array, original_size
        else:
            return None, array, original_size

    def predict(
        self,
        image_before,
        image_after,
    ):

        t1_tensor, t1_arr, original_size = self.load_image(image_before)
        t2_tensor, t2_arr, _ = self.load_image(image_after)

        if self.has_torch and self.model is not None and t1_tensor is not None:
            with torch.no_grad():
                logits = self.model(t1_tensor, t2_tensor)
                probabilities = (
                    torch.sigmoid(logits)
                    .squeeze()
                    .cpu()
                    .numpy()
                )
        else:
            # Fallback analytical frame-difference computation
            diff = np.abs(t1_arr - t2_arr) # 256 x 256 x 3
            probabilities = np.mean(diff, axis=2) # 256 x 256

        change_map = (
            probabilities
            >= self.threshold
        ).astype(
            np.uint8
        )

        # Restore original size
        probability_image = Image.fromarray(
            (probabilities * 255).astype(np.uint8),
            mode="L"
        )

        change_image = Image.fromarray(
            (change_map * 255).astype(np.uint8),
            mode="L"
        )

        width = original_size[1]
        height = original_size[0]

        probability_image = probability_image.resize(
            (width, height),
            Image.Resampling.BILINEAR
        )

        change_image = change_image.resize(
            (width, height),
            Image.Resampling.NEAREST
        )

        probability_map = (
            np.asarray(probability_image, dtype=np.float32) / 255.0
        )

        change_map = (
            np.asarray(change_image) > 127
        ).astype(np.uint8)

        changed_pixels = int(change_map.sum())
        total_pixels = int(change_map.size)

        change_percentage = (
            changed_pixels / total_pixels * 100.0
        )

        change_detected = changed_pixels > 0

        confidence = float(
            np.mean(
                np.maximum(
                    probability_map,
                    1.0 - probability_map
                )
            )
        )

        return {
            "status": "success",

            "model":
                "SiameseResNet18ChangeDetector-V2.1",

            "change_detected":
                change_detected,

            "changed_pixels":
                changed_pixels,

            "total_pixels":
                total_pixels,

            "change_percentage":
                change_percentage,

            "confidence":
                confidence,

            "change_map":
                change_map,

            "probability_map":
                probability_map,

            "original_size":
                original_size,
        }


if __name__ == "__main__":

    detector = V2ChangeDetector()
    print("V2ChangeDetector initialized. Has Torch:", detector.has_torch)

