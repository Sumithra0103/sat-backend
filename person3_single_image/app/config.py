"""Configuration module for SatQuery AI - Person 3 Single Image Module.

Manages environment flags and model parameters with safe mock defaults.
"""

import os
from dataclasses import dataclass


def _get_bool(env_var: str, default: bool = False) -> bool:
    val = os.getenv(env_var)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "t", "yes", "y")


@dataclass
class Settings:
    """Application and model runtime settings."""

    # Real model toggle: MUST DEFAULT TO FALSE
    use_real_rsllava: bool = _get_bool("SATQUERY_USE_REAL_RSLLAVA", False)

    # RS-LLaVA Model identifiers
    model_name: str = os.getenv("RSLLAVA_MODEL_NAME", "BigData-KSU/RS-llava-v1.5-7b-LoRA")
    model_base: str = os.getenv("RSLLAVA_MODEL_BASE", "Intel/neural-chat-7b-v3-3")

    # Hardware & Precision settings
    device: str = os.getenv("RSLLAVA_DEVICE", "auto")
    torch_dtype: str = os.getenv("RSLLAVA_TORCH_DTYPE", "float16")

    # Quantization: MUST DEFAULT TO FALSE (strictly optional)
    load_in_4bit: bool = _get_bool("RSLLAVA_LOAD_IN_4BIT", False)
    load_in_8bit: bool = _get_bool("RSLLAVA_LOAD_IN_8BIT", False)


def get_settings() -> Settings:
    """Returns fresh settings instance reading current environment."""
    return Settings(
        use_real_rsllava=_get_bool("SATQUERY_USE_REAL_RSLLAVA", False),
        model_name=os.getenv("RSLLAVA_MODEL_NAME", "BigData-KSU/RS-llava-v1.5-7b-LoRA"),
        model_base=os.getenv("RSLLAVA_MODEL_BASE", "Intel/neural-chat-7b-v3-3"),
        device=os.getenv("RSLLAVA_DEVICE", "auto"),
        torch_dtype=os.getenv("RSLLAVA_TORCH_DTYPE", "float16"),
        load_in_4bit=_get_bool("RSLLAVA_LOAD_IN_4BIT", False),
        load_in_8bit=_get_bool("RSLLAVA_LOAD_IN_8BIT", False),
    )


# Global default settings
settings = get_settings()
