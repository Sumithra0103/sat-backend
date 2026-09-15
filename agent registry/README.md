# SatQuery AI - Agent Registry

SatQuery AI is an agentic vision-language assistant designed for single, bi-temporal, and cross-modal remote sensing satellite imagery analysis. The **Agent Registry** is the central repository and discovery service where domain-adapted remote sensing specialist agents register their capabilities, input modalities, parameters, health status, and endpoints.

---

## 🌟 Workflow Implementation (Steps 1 - 15)

```
                         START
                           │
                           ▼
                1. Define Agent Schema (schemas/agent_schema.py)
                           │
                           ▼
                2. Create Mock Agents (agents/mock_agents.py)
                           │
                           ▼
                3. Set Up Database (database/db.py & database/models.py)
                           │
                           ▼
              4. Agent Registration API (POST /agents/register)
                           │
                           ▼
                 5. Validate Metadata (services/registry_service.py)
                 ├── Required fields validation
                 ├── Unique agent ID enforcement
                 └── Valid parameter schemas & URLs
                           │
                           ▼
                  6. Store Agent (SQLite persistence)
                           │
                           ▼
                7. Agent Discovery API (GET /agents)
                           │
                           ▼
              8. Capability-Based Search (GET /agents?capability=...)
                           │
                           ▼
                9. Agent Details API (GET /agents/{agent_id})
                           │
                           ▼
               10. Health & Status Check (POST /agents/{agent_id}/health)
                           │
                           ▼
              11. Update / Deregister (PUT /agents/{id} & DELETE /agents/{id})
                           │
                           ▼
                 12. Error Handling (400, 404, 409, 401, 403, 500 error schemas)
                           │
                           ▼
              13. Authentication / Security (Header X-API-Key middleware)
                           │
                           ▼
                14. Test All APIs (pytest tests/test_agent_registry.py)
                           │
                           ▼
                15. Connect with Router (client/registry_client.py)
                           │
                           ▼
                         END
```

---

## 🛰️ Registered Remote Sensing Specialist Agents

| Agent ID | Specialist Name | Capabilities | Supported Modalities | Target Benchmarks / Datasets |
| :--- | :--- | :--- | :--- | :--- |
| `rs-vqa-agent` | SatQuery RS-VQA Specialist | Single-Image VQA, Scene Captioning | Optical, SAR | BigEarthNet.txt, RSVQA |
| `rs-grounding-agent` | Land-Cover Region Grounding Agent | Text-Guided Grounding, Land-Cover Captioning | Optical, SAR | VRSBench, BigEarthNet-MM |
| `rs-change-agent` | Bi-Temporal Change Detection & Change-VQA | Change Detection, Change-VQA, Spatial Change Maps | Bi-temporal Optical, Bi-temporal SAR | CDVQA, LEVIR-CD, ISRO/SAC |
| `rs-crossmodal-agent` | Optical-SAR Cross-Modal Fusion Agent | Cross-Modal Joint Feature Fusion, Cloud-Resilient VQA | Co-registered Optical + SAR | Cartosat-2S + RISAT SAR (ISRO/SAC) |

---

## ⚙️ REST API Reference

All endpoints return standardized JSON schemas and handle security via `X-API-Key`.

### 1. Agent Registration (`POST /agents/register`)
- **Headers**: `X-API-Key: satquery-admin-key-2026`
- **Request Body**:
```json
{
  "agent_id": "rs-vqa-agent",
  "name": "SatQuery RS-VQA Specialist",
  "version": "1.2.0",
  "description": "Domain-adapted remote sensing visual question answering agent.",
  "capabilities": ["single_image_vqa", "scene_captioning"],
  "input_modalities": ["optical", "sar"],
  "supported_formats": ["geotiff", "png", "jpeg"],
  "endpoint_url": "http://localhost:8001/v1/vqa/execute",
  "parameters_schema": {
    "confidence_threshold": 0.7,
    "max_tokens": 256
  },
  "metadata": {
    "training_dataset": "BigEarthNet.txt, RSVQA"
  }
}
```

### 2. Agent Discovery & Capability Search (`GET /agents`)
- **Headers**: `X-API-Key: satquery-router-key-2026` (optional for public read)
- **Query Parameters**:
  - `capability`: e.g. `bitemporal_change_detection`, `single_image_vqa`, `cross_modal_fusion`
  - `modality`: e.g. `optical`, `sar`, `bitemporal_optical`, `cross_modal_optical_sar`
  - `format`: e.g. `geotiff`, `tiff`, `png`, `jpeg`
  - `status`: e.g. `active`, `degraded`, `inactive`

### 3. Agent Details (`GET /agents/{agent_id}`)
- **Headers**: `X-API-Key: satquery-router-key-2026`

### 4. Health Telemetry (`POST /agents/{agent_id}/health`)
- **Headers**: `X-API-Key: satquery-router-key-2026`
- **Request Body**:
```json
{
  "status": "active",
  "metrics": {
    "latency_ms": 110,
    "gpu_memory_used_mb": 3800
  },
  "message": "Model active on CUDA device 0"
}
```

### 5. Update Agent (`PUT /agents/{agent_id}`)
- **Headers**: `X-API-Key: satquery-admin-key-2026`

### 6. Deregister Agent (`DELETE /agents/{agent_id}`)
- **Headers**: `X-API-Key: satquery-admin-key-2026`

---

## 🚀 Router Connection Client (`client/registry_client.py`)

The SatQuery Router imports `AgentRegistryClient` to perform query-driven discovery:

```python
from client.registry_client import AgentRegistryClient
from schemas.agent_schema import AgentCapability, InputModality

client = AgentRegistryClient(db_session=db)

# Router discovers matching change detection agent for bi-temporal query
agents = client.discover_agents(
    capability=AgentCapability.BITEMPORAL_CHANGE_DETECTION,
    modality=InputModality.BITEMPORAL_OPTICAL
)

selected_agent = agents[0]
print(f"Routed Query to: {selected_agent.name} at {selected_agent.endpoint_url}")
```

---

## 🧪 Testing & Demonstration

### Run End-to-End Workflow Demo:
```bash
py demo_registry.py
```

### Run Automated Pytest Suite:
```bash
py -m pytest tests/test_agent_registry.py -v
```

### Run FastAPI REST Server:
```bash
py main.py
```
Open interactive documentation at: `http://localhost:8000/docs`
