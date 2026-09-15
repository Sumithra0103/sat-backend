"""Image loader and preprocessor for remote sensing images.

Supports: JPG, JPEG, PNG, TIFF, TIF.
Safely validates file existence and extension, loads image without destructive resizing,
and ensures standard RGB format.
"""

from io import BytesIO
from pathlib import Path
from typing import Set, Union, Optional
from PIL import Image


SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}


class ImagePreprocessingError(Exception):
    """Base exception for image preprocessing errors."""
    pass


class ImageNotFoundError(ImagePreprocessingError):
    """Raised when the specified image file does not exist."""
    pass


class UnsupportedImageFormatError(ImagePreprocessingError):
    """Raised when an unsupported image extension or format is provided."""
    pass


class ImageLoadError(ImagePreprocessingError):
    """Raised when an image cannot be decoded or opened."""
    pass


def validate_image_path(image_path: Union[str, Path]) -> Path:
    """Validates that the image file exists and has a supported extension.

    Args:
        image_path: Path to the image file (string or Path object).

    Returns:
        Validated pathlib.Path object.

    Raises:
        ImageNotFoundError: If the file does not exist or is a directory.
        UnsupportedImageFormatError: If the extension is not in SUPPORTED_EXTENSIONS.
    """
    path = Path(image_path).resolve()

    if not path.exists():
        raise ImageNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        raise ImageNotFoundError(f"Specified path is not a file: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedImageFormatError(
            f"Unsupported image extension '{ext}'. Supported extensions are: {supported_str}"
        )

    return path


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Safely loads a remote sensing image from a file path.

    - Validates file existence and supported extension (.jpg, .jpeg, .png, .tiff, .tif).
    - Preserves original image dimensions (no blind resizing).
    - Converts to standard RGB color mode.

    Args:
        image_path: Path to the target image file.

    Returns:
        Pillow Image object in RGB mode.

    Raises:
        ImageNotFoundError: If the image file does not exist.
        UnsupportedImageFormatError: If the file extension is not supported.
        ImageLoadError: If Pillow cannot read/decode the file.
    """
    path = validate_image_path(image_path)

    try:
        with Image.open(path) as img:
            # Fully load image data into memory so the file handle can be safely closed
            img.load()
            # Convert to RGB if needed (e.g. RGBA, Grayscale, CMYK) while preserving dimensions
            if img.mode != "RGB":
                rgb_img = img.convert("RGB")
            else:
                rgb_img = img.copy()
            return rgb_img
    except (ImageNotFoundError, UnsupportedImageFormatError):
        raise
    except Exception as exc:
        raise ImageLoadError(f"Failed to load image at {path}: {exc}") from exc


def load_image_from_bytes(data: bytes, filename: Optional[str] = None) -> Image.Image:
    """Safely loads an image from byte stream (e.g., from FastAPI UploadFile).

    Args:
        data: Raw image bytes.
        filename: Optional filename for extension validation.

    Returns:
        Pillow Image object in RGB mode.

    Raises:
        UnsupportedImageFormatError: If filename extension is unsupported.
        ImageLoadError: If bytes cannot be decoded as an image.
    """
    if not data:
        raise ImageLoadError("Received empty image data.")

    if filename:
        ext = Path(filename).suffix.lower()
        if ext and ext not in SUPPORTED_EXTENSIONS:
            supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise UnsupportedImageFormatError(
                f"Unsupported image extension '{ext}'. Supported extensions are: {supported_str}"
            )

    try:
        with Image.open(BytesIO(data)) as img:
            img.load()
            if img.mode != "RGB":
                return img.convert("RGB")
            return img.copy()
    except UnsupportedImageFormatError:
        raise
    except Exception as exc:
        raise ImageLoadError(f"Failed to decode image from bytes: {exc}") from exc
