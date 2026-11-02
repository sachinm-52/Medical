"""
End-to-End Test for MedVision AI using real server & requests.
"""

import sys
import time
import threading
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from backend.app.main import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")

def test_full_pipeline():
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # Give server time to bind and run lifespan

    session = requests.Session()
    session.trust_env = False
    base_url = "http://127.0.0.1:8001"

    print("\n=======================================================")
    print(" Running MedVision AI End-to-End Verification Tests")
    print("=======================================================\n")

    # 1. Health check
    print("[1/8] Testing GET /api/health...")
    res = session.get(f"{base_url}/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health_data = res.json()
    print(f"      Status: {health_data['status']}, Device: {health_data['device']}")

    # 2. Models list
    print("\n[2/8] Testing GET /api/models...")
    res = session.get(f"{base_url}/api/models")
    assert res.status_code == 200, f"List models failed: {res.text}"
    models = res.json()
    ready_models = [m for m in models if m["status"] == "ready"]
    print(f"      Total models: {len(models)}, Ready models: {len(ready_models)}")
    for m in ready_models:
        print(f"      - {m['model_name']} ({m['status']})")
    assert len(ready_models) > 0, "At least one model must be ready!"

    # 3. Datasets list
    print("\n[3/8] Testing GET /api/datasets...")
    res = session.get(f"{base_url}/api/datasets")
    assert res.status_code == 200
    datasets = res.json()
    print(f"      Verified {len(datasets)} strict dataset specifications.")
    assert len(datasets) == 4

    # 4. Upload / Sample Inference
    print("\n[4/8] Testing POST /api/predict with Cardiomegaly sample X-ray...")
    target_model = ready_models[0]
    sample_img = "sample_cardiomegaly.png"

    pred_res = session.post(
        f"{base_url}/api/predict",
        data={
            "model_id": target_model["id"],
            "image_filename": sample_img,
            "threshold": 0.50
        }
    )
    assert pred_res.status_code == 200, f"Prediction failed: {pred_res.text}"
    pred_data = pred_res.json()
    analysis_id = pred_data["analysis_id"]
    print(f"      Analysis #{analysis_id} created successfully!")
    print(f"      Overall Assessment: {pred_data['overall_assessment']}")
    print(f"      Processing Latency: {pred_data['processing_time_ms']} ms")
    print(f"      Initial Grad-CAM Overlay: {pred_data['gradcam_url']}")

    # 5. Grad-CAM generation
    print("\n[5/8] Testing POST /api/gradcam for targeted pathology...")
    grad_res = session.post(
        f"{base_url}/api/gradcam",
        json={
            "analysis_id": analysis_id,
            "target_class_idx": 1,  # Cardiomegaly
            "alpha": 0.55
        }
    )
    assert grad_res.status_code == 200, f"Grad-CAM failed: {grad_res.text}"
    grad_data = grad_res.json()
    print(f"      Target: {grad_data['target_disease']}")
    print(f"      Heatmap URL: {grad_data['gradcam_heatmap_url']}")
    print(f"      Overlay URL: {grad_data['gradcam_overlay_url']}")

    # 6. PDF Report generation
    print("\n[6/8] Testing GET /api/reports/{id}/pdf...")
    pdf_res = session.get(f"{base_url}/api/reports/{analysis_id}/pdf")
    assert pdf_res.status_code == 200, f"PDF generation failed: {pdf_res.text}"
    assert pdf_res.headers["content-type"] == "application/pdf"
    print(f"      PDF report generated ({len(pdf_res.content)} bytes).")

    # 7. History retrieval
    print("\n[7/8] Testing GET /api/history...")
    hist_res = session.get(f"{base_url}/api/history")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    print(f"      Retrieved {hist_data['total']} total analysis records from database.")
    assert hist_data["total"] >= 1

    # 8. Model metrics
    print("\n[8/8] Testing GET /api/metrics...")
    met_res = session.get(f"{base_url}/api/metrics")
    assert met_res.status_code == 200
    metrics = met_res.json()
    print(f"      Retrieved {len(metrics)} experimental metrics from database.")

    print("\n=======================================================")
    print(" ALL 8 END-TO-END VERIFICATION CHECKS PASSED!")
    print("=======================================================\n")

if __name__ == "__main__":
    test_full_pipeline()
