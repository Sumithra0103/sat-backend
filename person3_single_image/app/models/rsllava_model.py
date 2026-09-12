"""Official RS-LLaVA model adapter for single-image remote sensing analysis.

Official Repository: https://github.com/BigData-KSU/RS-LLaVA
Official LoRA Model: BigData-KSU/RS-llava-v1.5-7b-LoRA
Official Base Model: Intel/neural-chat-7b-v3-3

Strictly lazy-loaded. Uses ONLY the official RS-LLaVA repository loader (load_pretrained_model).
Instantiating this class or importing this module DOES NOT download or load 7B model weights.
"""

from typing import Dict, Any, Optional
from PIL import Image

from app.config import get_settings


class RSLLaVAModel:
    """Official model adapter for RS-LLaVA (BigData-KSU/RS-llava-v1.5-7b-LoRA).

    Provides:
    - Strict lazy loading (weights are only loaded upon explicit inference)
    - Uses ONLY the official RS-LLaVA repository builder (from llava.model.builder import load_pretrained_model)
    - Remote-sensing VQA (image + question -> answer)
    - Remote-sensing captioning (image -> descriptive caption)
    - Conversation template: 'llava_v1'
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_base: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        load_in_4bit: Optional[bool] = None,
        load_in_8bit: Optional[bool] = None,
    ):
        settings = get_settings()
        self.model_path = model_path or settings.model_name
        self.model_base = model_base or settings.model_base
        self.device = device or settings.device
        self.torch_dtype = torch_dtype or settings.torch_dtype
        self.load_in_4bit = load_in_4bit if load_in_4bit is not None else settings.load_in_4bit
        self.load_in_8bit = load_in_8bit if load_in_8bit is not None else settings.load_in_8bit

        # Model handles remain None until explicit inference call
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None

    @property
    def model_name(self) -> str:
        return f"RS-LLaVA ({self.model_path})"

    def is_loaded(self) -> bool:
        """Returns True only if model weights are currently loaded in memory."""
        return self.model is not None

    def _ensure_loaded(self) -> None:
        """Loads RS-LLaVA weights using ONLY the official repository builder.

        Only executed when answer() or generate() is invoked.
        If the official 'llava' package is not installed, raises a clear RuntimeError.
        """
        if self.is_loaded():
            return

        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except ImportError as exc:
            raise RuntimeError(
                "Official RS-LLaVA package ('llava') is not installed.\n"
                "RS-LLaVA requires the official repository implementation from:\n"
                "  https://github.com/BigData-KSU/RS-LLaVA\n"
                "To enable real inference on a GPU machine:\n"
                "  1. Clone the official repo: git clone https://github.com/BigData-KSU/RS-LLaVA.git\n"
                "  2. Install it in your environment: pip install -e RS-LLaVA\n"
                "  3. Install optional dependencies: pip install -r requirements-rsllava.txt\n"
                "Generic Transformers fallback is intentionally disabled to ensure authentic RS-LLaVA inference."
            ) from exc

        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=self.model_path,
            model_base=self.model_base,
            model_name=model_name,
            load_4bit=self.load_in_4bit,
            load_8bit=self.load_in_8bit,
            device=self.device,
        )

    def _run_inference(self, image: Image.Image, prompt_text: str) -> str:
        """Executes multimodal inference using official RS-LLaVA conversation templates."""
        self._ensure_loaded()
        import torch
        from llava.conversation import conv_templates
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.mm_utils import tokenizer_image_token

        # Official conversation mode: llava_v1
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        if self.device != "auto" and hasattr(input_ids, "to"):
            target_device = self.device if self.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            input_ids = input_ids.to(target_device)

        if hasattr(self.image_processor, "preprocess"):
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image_tensor = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]

        if hasattr(image_tensor, "unsqueeze"):
            image_tensor = image_tensor.unsqueeze(0)
        if self.device != "auto" and hasattr(image_tensor, "to"):
            target_device = self.device if self.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            image_tensor = image_tensor.to(target_device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=image_tensor,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
            )

        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        if conv.roles[1] in output_text:
            return output_text.split(conv.roles[1])[-1].strip()
        return output_text.strip()

    def answer(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Answers a remote sensing question using RS-LLaVA."""
        answer_text = self._run_inference(image, question)
        return {
            "answer": answer_text,
            "confidence": None,  # NEVER invent confidence values
            "evidence": [],
        }

    def generate(self, image: Image.Image, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generates a caption describing the remote sensing scene using RS-LLaVA."""
        caption_prompt = prompt or "Describe this remote sensing image in detail."
        caption_text = self._run_inference(image, caption_prompt)
        return {
            "caption": caption_text,
            "confidence": None,  # NEVER invent confidence values
        }
