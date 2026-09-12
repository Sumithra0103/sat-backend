# SatQuery AI — Person 3: Real RS-LLaVA Setup & Execution Guide

This guide documents the exact setup and execution procedure for running the **real RS-LLaVA (7B parameter)** model on a machine with sufficient GPU VRAM.

---

## 🖥️ System & Hardware Requirements

| Parameter | Requirement |
| :--- | :--- |
| **Model** | `BigData-KSU/RS-llava-v1.5-7b-LoRA` (LoRA weights) |
| **Base Model** | `Intel/neural-chat-7b-v3-3` |
| **Architecture** | LLaVA-v1.5 with conversation mode `llava_v1` |
| **GPU VRAM (FP16)** | $\ge$ 16 GB VRAM (e.g., RTX 3090, RTX 4090, A5000, A100) |
| **GPU VRAM (4-bit)**| $\ge$ 8 GB VRAM (e.g., RTX 3070, RTX 4060 Ti) with `bitsandbytes` |
| **OS** | Linux (Ubuntu 20.04/22.04 recommended) or Windows with CUDA 11.8+ |
| **Python** | Python 3.10+ |

> [!WARNING]
> Do NOT attempt to run real 7B inference on laptops with 4GB VRAM (such as NVIDIA GTX 1650). 
> The GTX 1650 is intended for development, API testing, preprocessing, and mock execution.

---

## 🛠️ Step-by-Step Setup on the GPU Machine

### Step 1: Create and Activate Virtual Environment
```bash
# On Linux:
python3 -m venv .venv
source .venv/bin/activate

# On Windows:
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install Base Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install Optional RS-LLaVA Dependencies
```bash
pip install -r requirements-rsllava.txt
```
*(Installs `torch`, `transformers==4.35`, `einops`, `SentencePiece`, `accelerate`, `peft`, `Pillow`).*

### Step 4: Install Official RS-LLaVA Repository
The adapter strictly requires the official repository implementation:
```bash
git clone https://github.com/BigData-KSU/RS-LLaVA.git
cd RS-LLaVA
pip install -e .
cd ..
```

---

## ⚙️ Enabling Real RS-LLaVA Mode

### On Windows PowerShell:
```powershell
# 1. Enable real model mode
$env:SATQUERY_USE_REAL_RSLLAVA="true"

# 2. (Optional) Enable 4-bit quantization if on 8GB-12GB VRAM
$env:RSLLAVA_LOAD_IN_4BIT="false"

# 3. Model identifiers (already defaulted in config)
$env:RSLLAVA_MODEL_NAME="BigData-KSU/RS-llava-v1.5-7b-LoRA"
$env:RSLLAVA_MODEL_BASE="Intel/neural-chat-7b-v3-3"
$env:RSLLAVA_DEVICE="cuda"
```

### On Linux / macOS:
```bash
# 1. Enable real model mode
export SATQUERY_USE_REAL_RSLLAVA=true

# 2. (Optional) Enable 4-bit quantization if on 8GB-12GB VRAM
export RSLLAVA_LOAD_IN_4BIT=false

# 3. Model identifiers
export RSLLAVA_MODEL_NAME="BigData-KSU/RS-llava-v1.5-7b-LoRA"
export RSLLAVA_MODEL_BASE="Intel/neural-chat-7b-v3-3"
export RSLLAVA_DEVICE="cuda"
```

---

## 🚀 Running the Server & Testing Real Inference

### 1. Start the FastAPI Server
```bash
# Port 8001 for Windows (or 8000 on Linux):
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

*Note on Lazy Loading:* The server will start in seconds without loading the 7B weights into GPU memory. The weights will only be loaded on the first actual `/vqa` or `/caption` call.

### 2. Verify `/health` Status
```bash
curl http://127.0.0.1:8001/health
```
Response will reflect:
```json
{
  "status": "ok",
  "module": "person3_single_image",
  "real_rsllava_enabled": true,
  "active_models": {
    "vqa": "RS-LLaVA (BigData-KSU/RS-llava-v1.5-7b-LoRA)",
    "caption": "RS-LLaVA (BigData-KSU/RS-llava-v1.5-7b-LoRA) (Captioning)",
    "grounding": "MockGrounding"
  }
}
```

### 3. Test Real VQA (`POST /vqa`)
```bash
curl -X POST "http://127.0.0.1:8001/vqa" \
  -F "image=@outputs/satellite_sample.tif" \
  -F "question=What structures or terrain features are present in this remote sensing tile?"
```

### 4. Test Real Captioning (`POST /caption`)
```bash
curl -X POST "http://127.0.0.1:8001/caption" \
  -F "image=@outputs/satellite_sample.tif"
```

### 5. Test Real Inference via CLI Script
```bash
python scripts/test_rsllava.py --run-real --image outputs/satellite_sample.tif --question "Describe the scene."
```

---

## 🔄 Reverting to Local Mock Mode
To switch back to safe, lightweight mock mode:
```powershell
# Windows
$env:SATQUERY_USE_REAL_RSLLAVA="false"

# Linux
export SATQUERY_USE_REAL_RSLLAVA=false
```
All endpoints will instantly revert to `MockVQA` and `MockCaption` with zero GPU requirements.
