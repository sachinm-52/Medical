"""
Pure In-Process End-to-End Verification Test for MedVision AI.
Tests every API function, database operation, neural network inference, Grad-CAM, and PDF generation directly.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database.session import SessionLocal, Base, engine
from backend.app.services.model_service import model_service
from backend.app.routes.api_routes import (
    get_health,
    list_models,
    list_datasets,
    run_prediction,
    compute_gradcam,
    get_history,
    download_pdf_report,
    get_model_metrics
)
from backend.app.schemas.api_schemas import GradCamRequest

def run_tests():
    print("\n=======================================================")
    print(" Running MedVision AI In-Process Verification Tests")
    print("=======================================================\n")

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # 1. Health check
        print("[1/8] Verifying get_health()...")
        health = get_health(db)
        print(f"      Status: {health.status}, Device: {health.device}")
        assert health.status == "operational"

        # 2. Models list
        print("\n[2/8] Verifying list_models()...")
        models = list_models(db)
        ready_models = [m for m in models if m.status == "ready"]
        print(f"      Registered: {len(models)} models, Ready: {len(ready_models)}")
        for m in ready_models:
            print(f"      - ID #{m.id}: {m.model_name} ({m.dataset_name})")
        assert len(ready_models) >= 3, "Expected at least 3 models with trained weights ready!"

        # 3. Datasets list
        print("\n[3/8] Verifying list_datasets()...")
        datasets = list_datasets()
        print(f"      Verified {len(datasets)} dataset specifications.")
        assert len(datasets) == 4

        # 4. Neural Network Inference
        print("\n[4/8] Verifying run_prediction() with Cardiomegaly sample X-ray...")
        target_model = ready_models[0]
        sample_img = "sample_cardiomegaly.png"
        pred = run_prediction(
            model_id=target_model.id,
            image_filename=sample_img,
            threshold=0.50,
            db=db
        )
        print(f"      Analysis ID: #{pred.analysis_id}")
        print(f"      Overall Assessment: {pred.overall_assessment}")
        print(f"      Processing Latency: {pred.processing_time_ms} ms")
        print(f"      Top Prediction: {pred.top_prediction.disease if pred.top_prediction else 'None'}")
        print(f"      Initial Grad-CAM Overlay: {pred.gradcam_url}")
        assert pred.analysis_id > 0
        assert len(pred.predictions) > 0

        # 5. Grad-CAM Interpretation
        print("\n[5/8] Verifying compute_gradcam() with transparency blending...")
        grad_req = GradCamRequest(
            analysis_id=pred.analysis_id,
            target_class_idx=1,
            alpha=0.60
        )
        grad_res = compute_gradcam(grad_req, db=db)
        print(f"      Target Disease: {grad_res.target_disease}")
        print(f"      Heatmap URL: {grad_res.gradcam_heatmap_url}")
        print(f"      Overlay URL: {grad_res.gradcam_overlay_url}")
        assert grad_res.gradcam_overlay_url is not None

        # 6. PDF Report Generation
        print("\n[6/8] Verifying download_pdf_report()...")
        pdf_response = download_pdf_report(pred.analysis_id, db=db)
        print(f"      Generated PDF Report at: {pdf_response.path}")
        assert Path(pdf_response.path).exists()
        assert Path(pdf_response.path).stat().st_size > 1000

        # 7. History Retrieval & Filtering
        print("\n[7/8] Verifying get_history()...")
        hist = get_history(limit=10, db=db)
        print(f"      Found {hist.total} historical analysis records in DB.")
        assert hist.total >= 1

        # 8. Model Evaluation Metrics
        print("\n[8/8] Verifying get_model_metrics()...")
        metrics = get_model_metrics(model_id=target_model.id, db=db)
        print(f"      Retrieved {len(metrics)} experimental metrics from DB.")
        for m in metrics[:5]:
            print(f"      - {m.metric_name}: {m.metric_value if m.metric_value is not None else 'Complex Data'}")
        assert len(metrics) > 0

    finally:
        db.close()

    print("\n=======================================================")
    print(" ALL 8 IN-PROCESS VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=======================================================\n")

if __name__ == "__main__":
    run_tests()
