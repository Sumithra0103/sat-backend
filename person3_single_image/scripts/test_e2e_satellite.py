"""End-to-end API test script using a real remote-sensing satellite image."""

import json
import httpx
from pathlib import Path

# Add repo root to python path
repo_root = Path(__file__).resolve().parent.parent

client = httpx.Client(base_url="http://127.0.0.1:8001", timeout=15.0)

image_path = repo_root / "outputs" / "satellite_sample.tif"
if not image_path.exists():
    raise FileNotFoundError(f"Sample satellite image not found at {image_path}")

with open(image_path, "rb") as f:
    img_bytes = f.read()

print("=" * 70)
print("SatQuery AI — End-to-End Satellite Image API Verification")
print(f"Target Image: {image_path.name} ({len(img_bytes)} bytes)")
print("Target Server: http://127.0.0.1:8001")
print("=" * 70)

# ----------------------------------------------------------------------
# 1. Test POST /vqa with 3 questions
# ----------------------------------------------------------------------
vqa_questions = [
    "What is present in this image?",
    "Are there buildings in this image?",
    "Is this area urban or rural?",
]

vqa_responses = []
for q in vqa_questions:
    print(f"\n[POST /vqa] Query: '{q}'")
    files = {"image": (image_path.name, img_bytes, "image/tiff")}
    data = {"question": q}
    resp = client.post("/vqa", files=files, data=data)
    print("  Status Code:", resp.status_code)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    res_json = resp.json()
    print("  Model:", res_json.get("model"))
    print("  Answer:", res_json.get("answer"))
    print("  Confidence:", res_json.get("confidence"))
    vqa_responses.append(res_json)

# Save to outputs/answers/
answers_dir = repo_root / "outputs" / "answers"
answers_dir.mkdir(parents=True, exist_ok=True)
with open(answers_dir / "vqa_e2e_results.json", "w", encoding="utf-8") as f:
    json.dump(vqa_responses, f, indent=2)

# ----------------------------------------------------------------------
# 2. Test POST /caption
# ----------------------------------------------------------------------
print("\n[POST /caption]")
files = {"image": (image_path.name, img_bytes, "image/tiff")}
resp_cap = client.post("/caption", files=files)
print("  Status Code:", resp_cap.status_code)
assert resp_cap.status_code == 200
cap_json = resp_cap.json()
print("  Model:", cap_json.get("model"))
print("  Caption:", cap_json.get("caption"))
print("  Confidence:", cap_json.get("confidence"))

# Save to outputs/captions/
captions_dir = repo_root / "outputs" / "captions"
captions_dir.mkdir(parents=True, exist_ok=True)
with open(captions_dir / "caption_e2e_result.json", "w", encoding="utf-8") as f:
    json.dump(cap_json, f, indent=2)

# ----------------------------------------------------------------------
# 3. Test POST /grounding
# ----------------------------------------------------------------------
print("\n[POST /grounding] Query: 'Locate the buildings in this image'")
files = {"image": (image_path.name, img_bytes, "image/tiff")}
data_grd = {"query": "Locate the buildings in this image"}
resp_grd = client.post("/grounding", files=files, data=data_grd)
print("  Status Code:", resp_grd.status_code)
assert resp_grd.status_code == 200
grd_json = resp_grd.json()
print("  Model:", grd_json.get("model"))
print("  Boxes:", grd_json.get("boxes"))
print("  Labels:", grd_json.get("labels"))
print("  Confidence:", grd_json.get("confidence"))

# ----------------------------------------------------------------------
# 4. Test POST /analyze
# ----------------------------------------------------------------------
print("\n[POST /analyze] Query: 'What is present in this image?'")
files = {"image": (image_path.name, img_bytes, "image/tiff")}
data_ana = {"query": "What is present in this image?"}
resp_ana = client.post("/analyze", files=files, data=data_ana)
print("  Status Code:", resp_ana.status_code)
assert resp_ana.status_code == 200
ana_json = resp_ana.json()
print("  Task Detected:", ana_json.get("task"))
print("  Answer:", ana_json.get("answer"))
evidence_list = ana_json.get("evidence", [])
print(f"  Evidence Generated: {len(evidence_list)} item(s)")
if evidence_list:
    ev_path = Path(evidence_list[0]["evidence_path"])
    print(f"  Saved Evidence Path: {ev_path}")
    assert ev_path.exists(), f"Evidence file does not exist: {ev_path}"

print("\n" + "=" * 70)
print("ALL ENDPOINTS COMPLETED END-TO-END VERIFICATION SUCCESSFULLY!")
print("=" * 70)
