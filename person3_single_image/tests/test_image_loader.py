"""Unit tests for image loading and preprocessing."""

import pytest
from pathlib import Path
from PIL import Image

from app.preprocessing.image_loader import (
    load_image,
    load_image_from_bytes,
    validate_image_path,
    ImageNotFoundError,
    UnsupportedImageFormatError,
    ImageLoadError,
)


@pytest.fixture
def sample_images(tmp_path: Path):
    """Creates temporary images in various formats and modes for testing."""
    # 1. Standard RGB PNG (320 x 240)
    png_path = tmp_path / "test_satellite.png"
    img_png = Image.new("RGB", (320, 240), color=(100, 150, 200))
    img_png.save(png_path)

    # 2. RGBA JPG/PNG
    rgba_path = tmp_path / "test_transparent.png"
    img_rgba = Image.new("RGBA", (150, 150), color=(50, 50, 50, 128))
    img_rgba.save(rgba_path)

    # 3. JPEG
    jpg_path = tmp_path / "test_aerial.jpg"
    img_jpg = Image.new("RGB", (400, 300), color=(60, 120, 60))
    img_jpg.save(jpg_path)

    # 4. TIFF
    tiff_path = tmp_path / "test_multispectral.tif"
    img_tiff = Image.new("RGB", (512, 512), color=(20, 40, 80))
    img_tiff.save(tiff_path)

    # 5. Unsupported file
    txt_path = tmp_path / "invalid_file.txt"
    txt_path.write_text("not an image")

    return {
        "png": png_path,
        "rgba": rgba_path,
        "jpg": jpg_path,
        "tif": tiff_path,
        "txt": txt_path,
    }


def test_valid_image_loading_preserves_dimensions(sample_images):
    """Verifies that images load correctly and dimensions are strictly preserved."""
    loaded = load_image(sample_images["png"])
    assert isinstance(loaded, Image.Image)
    assert loaded.size == (320, 240)
    assert loaded.mode == "RGB"

    loaded_jpg = load_image(sample_images["jpg"])
    assert loaded_jpg.size == (400, 300)
    assert loaded_jpg.mode == "RGB"

    loaded_tif = load_image(sample_images["tif"])
    assert loaded_tif.size == (512, 512)
    assert loaded_tif.mode == "RGB"


def test_rgba_converted_to_rgb(sample_images):
    """Verifies that RGBA images are converted to RGB mode."""
    loaded = load_image(sample_images["rgba"])
    assert loaded.mode == "RGB"
    assert loaded.size == (150, 150)


def test_invalid_image_path_raises_not_found(tmp_path):
    """Verifies that non-existent image paths raise ImageNotFoundError."""
    missing_path = tmp_path / "does_not_exist.png"
    with pytest.raises(ImageNotFoundError):
        load_image(missing_path)


def test_unsupported_extension_raises_error(sample_images):
    """Verifies that unsupported file extensions raise UnsupportedImageFormatError."""
    with pytest.raises(UnsupportedImageFormatError):
        load_image(sample_images["txt"])


def test_load_image_from_bytes(sample_images):
    """Verifies loading image directly from bytes."""
    with open(sample_images["png"], "rb") as f:
        data = f.read()

    loaded = load_image_from_bytes(data, filename="sample.png")
    assert loaded.size == (320, 240)
    assert loaded.mode == "RGB"


def test_load_image_from_empty_bytes():
    """Verifies error handling on empty byte streams."""
    with pytest.raises(ImageLoadError):
        load_image_from_bytes(b"")
