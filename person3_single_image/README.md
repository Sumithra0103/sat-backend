# SatQuery AI — Person 3: Single Image Remote-Sensing Module

Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries (SIH 2026).

---

## 📌 Module Responsibilities (Person 3)

1. **Single-Image Remote-Sensing VQA**: Answer natural language questions about aerial and satellite scenes.
2. **Single-Image Captioning**: Generate rich contextual descriptions of terrain, infrastructure, and land cover.
3. **Text-Guided Visual Grounding Interface**: Ground text queries to region-level visual targets.
4. **Visual Evidence Generation**: Produce verifiable visual evidence (bounding boxes or annotated Q&A panels) saved to `outputs/evidence/`.
5. **Image Preprocessing**: Safely load and validate remote sensing formats (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`) while preserving native spatial resolution without destructive blind resizing.
6. **Clean Model Abstractions**: Modular interfaces supporting both lightweight mock execution and real RS-LLaVA integration.

---

## ⚙️ Architecture & Model Integration Strategy

> [!NOTE]
> This module is configured to run with **lightweight mock models** by default (`SATQUERY_USE_REAL_RSLLAVA=false`).
> No 7B weights or heavy datasets are downloaded locally, enabling fast development and automated testing on standard laptops (e.g., GTX 1650 4GB VRAM, 8GB RAM).

### Model Architecture:
```
                    FastAPI Endpoints
             (/vqa, /caption, /grounding, /analyze)
                           │
                           ▼
                    Service Layer
          (VQAService, CaptionService, GroundingService)
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      Mock Implementations       Real Model Adapters
      • MockVQA                  • RSLLaVAVQA (BigData-KSU/RS-llava-v1.5-7b-LoRA)
      • MockCaption              • RSLLaVACaption (BigData-KSU/RS-llava-v1.5-7b-LoRA)
      • MockGrounding            • GeoChat (Future integration)
```

| Task | Mock Mode (Default) | Real Mode (`SATQUERY_USE_REAL_RSLLAVA=true`) |
| :--- | :--- | :--- |
| **VQA** | `MockVQA` | `RSLLaVAVQA` (`BigData-KSU/RS-llava-v1.5-7b-LoRA`) |
| **Captioning** | `MockCaption` | `RSLLaVACaption` (`BigData-KSU/RS-llava-v1.5-7b-LoRA`) |
| **Visual Grounding** | `MockGrounding` | `MockGrounding` (GeoChat planned for later) |

---

## 🧠 REAL RS-LLaVA INTEGRATION

The project includes a production-ready model adapter for **`BigData-KSU/RS-llava-v1.5-7b-LoRA`** (base model: `Intel/neural-chat-7b-v3-3`) located in `app/models/rsllava_model.py`.

### ⚠️ Hardware Reality & Low-VRAM Guidance
- **RS-LLaVA is a 7-Billion parameter multimodal model.**
- Running full 7B inference requires a suitable machine with high GPU VRAM (e.g., $\ge$16 GB VRAM for 16-bit float, or $\ge$8 GB VRAM for 4-bit quantized mode).
- **Your GTX 1650 laptop (4GB VRAM, 8GB RAM) is intended for local development, API building, image preprocessing, evidence generation, and mock verification.**
- Weights are **NEVER downloaded automatically** on startup, test runs, or health checks.

### Enabling Real RS-LLaVA Mode (When Running on a Capable Machine)

1. **Install the optional dependencies:**
   ```powershell
   pip install -r requirements-rsllava.txt
   ```

2. **Set the environment variable in PowerShell:**
   ```powershell
   # Enable real model mode
   $env:SATQUERY_USE_REAL_RSLLAVA="true"

   # (Optional) Customize device or base model
   $env:RSLLAVA_MODEL_NAME="BigData-KSU/RS-llava-v1.5-7b-LoRA"
   $env:RSLLAVA_MODEL_BASE="Intel/neural-chat-7b-v3-3"
   $env:RSLLAVA_DEVICE="cuda"

   # (Optional) 4-bit quantization (strictly optional, requires bitsandbytes)
   $env:RSLLAVA_LOAD_IN_4BIT="false"
   ```

3. **Run the Safe RS-LLaVA Verification Script:**
   ```powershell
   # Safe inspection (does NOT download weights)
   python scripts/test_rsllava.py

   # Explicit inference execution (requires --run-real flag)
   python scripts/test_rsllava.py --run-real
   ```

---

## 🚀 Quickstart Guide (Default Mock Mode on Windows)

Open PowerShell inside the `person3_single_image` directory:

```powershell
# 1. Create a Python virtual environment
python -m venv .venv

# 2. Activate the virtual environment
.venv\Scripts\activate

# 3. Install required lightweight dependencies
pip install -r requirements.txt
```

### Run Automated Tests (GPU NOT Required)
```powershell
pytest tests -v
```

### Start the FastAPI Server
```powershell
uvicorn app.main:app --reload
```

Server will start at: `http://127.0.0.1:8000`

---

## 📖 Interactive Swagger API Documentation

Once the server is running, open your web browser to:
👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## 🔌 API Endpoints Summary

| Method | Endpoint | Description | Input |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Module overview and endpoint index | None |
| `GET` | `/health` | Health check & active model backends | None |
| `POST` | `/vqa` | Single-image Visual Question Answering | `image` (file), `question` (text) |
| `POST` | `/caption` | Single-image Captioning | `image` (file) |
| `POST` | `/grounding` | Text-guided visual grounding | `image` (file), `query` (text) |
| `POST` | `/analyze` | Unified pipeline with keyword auto-detection | `image` (file), `query` (text), `task` (optional) |

### Sample Response (`POST /vqa`):
```json
{
  "success": true,
  "task": "single_image_vqa",
  "question": "What is the primary land use?",
  "answer": "[TEST MOCK ANSWER] Remote sensing scene analyzed (256x256 px). Query: 'What is the primary land use?'. Pipeline functional.",
  "model": "MockVQA",
  "confidence": null,
  "evidence": []
}
```

---

## 🧪 CLI Test Scripts

```powershell
# Test image loading and preprocessing
python scripts/test_image.py

# Test VQA service
python scripts/test_vqa.py

# Test Captioning service
python scripts/test_caption.py

# Test Visual Grounding service
python scripts/test_grounding.py

# Test Unified Pipeline (all tasks + evidence generation)
python scripts/test_pipeline.py

# Test RS-LLaVA configuration and safety guards
python scripts/test_rsllava.py
```

Outputs are saved under:
- `outputs/answers/`
- `outputs/captions/`
- `outputs/evidence/`
