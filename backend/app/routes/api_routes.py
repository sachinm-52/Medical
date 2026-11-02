"""
Main API Routes for MedVision AI.
Implements endpoints for Upload, Inference, Grad-CAM, History, Datasets, and Metrics.
"""

import os
import uuid
import datetime
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import torch

from backend.app.config import settings
from backend.app.database.session import get_db
from backend.app.models.schema_models import ModelRecord, AnalysisRecord, ModelMetric
from backend.app.schemas.api_schemas import (
    ModelSummary,
    DatasetInfo,
    PredictionResponse,
    GradCamRequest,
    GradCamResponse,
    HistoryListResponse,
    HistoryItem,
    MetricItem,
    HealthResponse
)
from backend.app.services.model_service import model_service
from backend.app.services.inference_service import inference_service
from backend.app.services.gradcam_service import gradcam_service
from backend.app.services.pdf_service import pdf_service

router = APIRouter()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

@router.get("/health", response_model=HealthResponse)
def get_health(db: Session = Depends(get_db)):
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    db_status = "connected"
    try:
        db.execute(torch.text("SELECT 1") if hasattr(torch, "text") else "SELECT 1")
    except Exception:
        db_status = "sqlite"

    return HealthResponse(
        status="operational",
        version="1.0.0",
        cuda_available=cuda_avail,
        device=device_name,
        database=db_status,
        timestamp=datetime.datetime.utcnow()
    )

@router.get("/models", response_model=List[ModelSummary])
def list_models(db: Session = Depends(get_db)):
    return model_service.get_models(db)

@router.get("/datasets", response_model=List[DatasetInfo])
def list_datasets():
    return model_service.get_datasets()

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and validate chest X-ray image (JPG, JPEG, PNG).
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Only JPG, JPEG, and PNG are allowed."
        )

    # Read contents and check size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum allowed size of 25MB."
        )

    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = settings.UPLOAD_DIR / unique_filename
    with open(save_path, "wb") as f:
        f.write(contents)

    rel_url = f"/api/uploads/{unique_filename}"
    return {
        "filename": unique_filename,
        "image_path": str(save_path),
        "image_url": rel_url,
        "file_size": len(contents)
    }

@router.post("/predict", response_model=PredictionResponse)
def run_prediction(
    model_id: int = Form(...),
    image_filename: str = Form(...),
    threshold: float = Form(0.5),
    db: Session = Depends(get_db)
):
    """
    Perform deep learning model inference and initial Grad-CAM explanation.
    """
    model_rec = model_service.get_model_by_id(db, model_id)
    if not model_rec:
        raise HTTPException(status_code=404, detail="Selected model not found.")

    if model_rec.status != "ready" or not model_rec.weights_path or not Path(model_rec.weights_path).exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_rec.model_name}' weights have not been trained yet. "
                "Per system policy, inference cannot be simulated or mocked without trained weights."
            )
        )

    image_path = settings.UPLOAD_DIR / image_filename
    if not image_path.exists():
        # Check sample dir
        sample_path = settings.SAMPLE_DIR / image_filename
        if sample_path.exists():
            image_path = sample_path
        else:
            raise HTTPException(status_code=404, detail=f"Image file '{image_filename}' not found.")

    # Load neural network model
    loaded_model = inference_service.load_model(
        architecture=model_rec.architecture,
        num_classes=model_rec.num_classes,
        task_type=model_rec.task_type,
        weights_path=model_rec.weights_path
    )

    # Run forward pass
    inference_result = inference_service.predict(
        image_path=str(image_path),
        model=loaded_model,
        class_labels=model_rec.class_labels,
        task_type=model_rec.task_type,
        threshold=threshold
    )

    # Automatically generate initial Grad-CAM for top finding or class 0
    top_pred = inference_result["top_prediction"]
    target_idx = 0
    if top_pred and model_rec.task_type == "multi_label":
        try:
            target_idx = model_rec.class_labels.index(top_pred.disease)
        except ValueError:
            target_idx = 0

    try:
        heatmap_url, overlay_url, overlay_path = gradcam_service.generate_gradcam(
            model=loaded_model,
            image_path=str(image_path),
            target_class_idx=target_idx,
            alpha=0.5
        )
    except Exception as e:
        print(f"Grad-CAM generation warning: {e}")
        overlay_url = None
        overlay_path = None

    # Save analysis record to database
    record = AnalysisRecord(
        model_id=model_rec.id,
        image_filename=image_filename,
        image_path=str(image_path),
        gradcam_path=overlay_path,
        prediction_results={
            "predictions": [p.dict() for p in inference_result["predictions"]],
            "top_prediction": top_pred.dict() if top_pred else None,
            "overall_assessment": inference_result["overall_assessment"],
            "processing_time_ms": inference_result["processing_time_ms"]
        },
        processing_time=inference_result["processing_time_ms"] / 1000.0,
        status="completed"
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    image_url = f"/api/uploads/{image_filename}" if (settings.UPLOAD_DIR / image_filename).exists() else f"/api/samples/{image_filename}"

    return PredictionResponse(
        analysis_id=record.id,
        image_filename=image_filename,
        image_url=image_url,
        gradcam_url=overlay_url,
        model_id=model_rec.id,
        model_name=model_rec.model_name,
        architecture=model_rec.architecture,
        dataset_name=model_rec.dataset_name,
        task_type=model_rec.task_type,
        predictions=inference_result["predictions"],
        top_prediction=top_pred,
        overall_assessment=inference_result["overall_assessment"],
        processing_time_ms=inference_result["processing_time_ms"],
        timestamp=record.created_at,
        disclaimer="FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY. Not a confirmed medical diagnosis."
    )

@router.post("/gradcam", response_model=GradCamResponse)
def compute_gradcam(req: GradCamRequest, db: Session = Depends(get_db)):
    """
    Generate customized Grad-CAM with specific target disease index and transparency alpha.
    """
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == req.analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis record not found.")

    model_rec = db.query(ModelRecord).filter(ModelRecord.id == record.model_id).first()
    if not model_rec or not model_rec.weights_path:
        raise HTTPException(status_code=400, detail="Associated model weights are unavailable.")

    loaded_model = inference_service.load_model(
        architecture=model_rec.architecture,
        num_classes=model_rec.num_classes,
        task_type=model_rec.task_type,
        weights_path=model_rec.weights_path
    )

    heatmap_url, overlay_url, overlay_path = gradcam_service.generate_gradcam(
        model=loaded_model,
        image_path=record.image_path,
        target_class_idx=req.target_class_idx,
        alpha=req.alpha
    )

    # Update record gradcam path
    record.gradcam_path = overlay_path
    db.commit()

    target_disease_name = "Overall / Top Finding"
    if req.target_class_idx is not None and model_rec.class_labels and req.target_class_idx < len(model_rec.class_labels):
        target_disease_name = model_rec.class_labels[req.target_class_idx]

    img_rel_url = f"/api/uploads/{record.image_filename}" if (settings.UPLOAD_DIR / record.image_filename).exists() else f"/api/samples/{record.image_filename}"

    return GradCamResponse(
        analysis_id=record.id,
        original_image_url=img_rel_url,
        gradcam_heatmap_url=heatmap_url,
        gradcam_overlay_url=overlay_url,
        target_disease=target_disease_name,
        explanation_note=(
            "Grad-CAM visualizes gradients of the target class score with respect to convolutional "
            "feature activations. High-intensity red regions contributed most to model activation; "
            "they are model explanations, not clinical confirmation of lesion boundaries."
        )
    )

@router.get("/history", response_model=HistoryListResponse)
def get_history(
    search: Optional[str] = None,
    model_id: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    query = db.query(AnalysisRecord)
    if model_id:
        query = query.filter(AnalysisRecord.model_id == model_id)
    if search:
        query = query.filter(AnalysisRecord.image_filename.ilike(f"%{search}%"))

    total = query.count()
    records = query.order_by(AnalysisRecord.created_at.desc()).offset(offset).limit(limit).all()

    items = []
    for r in records:
        pred_res = r.prediction_results or {}
        img_url = f"/api/uploads/{r.image_filename}" if (settings.UPLOAD_DIR / r.image_filename).exists() else f"/api/samples/{r.image_filename}"
        gradcam_url = None
        if r.gradcam_path and Path(r.gradcam_path).exists():
            gradcam_url = f"/api/storage/gradcam/{Path(r.gradcam_path).name}"

        items.append(HistoryItem(
            id=r.id,
            model_name=r.model.model_name if r.model else "Unknown",
            dataset_name=r.model.dataset_name if r.model else "Unknown",
            image_filename=r.image_filename,
            image_url=img_url,
            gradcam_url=gradcam_url,
            predictions=pred_res.get("predictions", []),
            top_prediction=pred_res.get("top_prediction"),
            overall_assessment=pred_res.get("overall_assessment", "Pending"),
            processing_time=r.processing_time,
            status=r.status,
            created_at=r.created_at
        ))

    return HistoryListResponse(total=total, items=items)

@router.get("/history/{analysis_id}")
def get_history_detail(analysis_id: int, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis record not found.")

    pred_res = record.prediction_results or {}
    img_url = f"/api/uploads/{record.image_filename}" if (settings.UPLOAD_DIR / record.image_filename).exists() else f"/api/samples/{record.image_filename}"
    gradcam_url = None
    if record.gradcam_path and Path(record.gradcam_path).exists():
        gradcam_url = f"/api/storage/gradcam/{Path(record.gradcam_path).name}"

    return {
        "id": record.id,
        "model_id": record.model_id,
        "model_name": record.model.model_name if record.model else "Unknown",
        "architecture": record.model.architecture if record.model else "Unknown",
        "dataset_name": record.model.dataset_name if record.model else "Unknown",
        "image_filename": record.image_filename,
        "image_url": img_url,
        "gradcam_url": gradcam_url,
        "predictions": pred_res.get("predictions", []),
        "top_prediction": pred_res.get("top_prediction"),
        "overall_assessment": pred_res.get("overall_assessment"),
        "processing_time": record.processing_time,
        "created_at": record.created_at
    }

@router.delete("/history/{analysis_id}")
def delete_history_item(analysis_id: int, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis record not found.")

    # Safely remove gradcam overlay file if it exists
    if record.gradcam_path and Path(record.gradcam_path).exists():
        try:
            os.remove(record.gradcam_path)
        except OSError:
            pass

    db.delete(record)
    db.commit()
    return {"message": f"Analysis record #{analysis_id} deleted successfully."}

@router.get("/metrics", response_model=List[MetricItem])
def get_model_metrics(model_id: Optional[int] = None, db: Session = Depends(get_db)):
    """
    Retrieve experimental evaluation metrics from real trained models.
    """
    # Check for evaluation_metrics.json in checkpoint directories and sync
    for model_rec in db.query(ModelRecord).all():
        if model_rec.weights_path:
            metrics_file = Path(model_rec.weights_path).parent / "evaluation_metrics.json"
            if metrics_file.exists():
                import json
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Update or insert ROC-AUC
                    for m_name in ["roc_auc", "pr_auc", "accuracy", "f1_score", "precision", "recall"]:
                        if m_name in data:
                            existing = db.query(ModelMetric).filter_by(model_id=model_rec.id, metric_name=m_name).first()
                            if not existing:
                                db.add(ModelMetric(
                                    model_id=model_rec.id,
                                    metric_name=m_name,
                                    metric_value=data[m_name],
                                    evaluation_split="test"
                                ))
                            else:
                                existing.metric_value = data[m_name]

                    # Per-class metrics
                    if "per_class" in data:
                        existing_pc = db.query(ModelMetric).filter_by(model_id=model_rec.id, metric_name="per_class").first()
                        if not existing_pc:
                            db.add(ModelMetric(
                                model_id=model_rec.id,
                                metric_name="per_class",
                                metric_data=data["per_class"],
                                evaluation_split="test"
                            ))
                        else:
                            existing_pc.metric_data = data["per_class"]

                    # Training curves
                    if "history" in data:
                        existing_hist = db.query(ModelMetric).filter_by(model_id=model_rec.id, metric_name="training_history").first()
                        if not existing_hist:
                            db.add(ModelMetric(
                                model_id=model_rec.id,
                                metric_name="training_history",
                                metric_data=data["history"],
                                evaluation_split="train_val"
                            ))
                        else:
                            existing_hist.metric_data = data["history"]

                    db.commit()
                except Exception as e:
                    print(f"Error syncing metrics for model {model_rec.id}: {e}")

    query = db.query(ModelMetric)
    if model_id:
        query = query.filter(ModelMetric.model_id == model_id)

    metrics = query.all()
    results = []
    for m in metrics:
        results.append(MetricItem(
            id=m.id,
            model_id=m.model_id,
            model_name=m.model.model_name if m.model else "Unknown",
            architecture=m.model.architecture if m.model else "Unknown",
            dataset_name=m.model.dataset_name if m.model else "Unknown",
            metric_name=m.metric_name,
            metric_value=m.metric_value,
            metric_data=m.metric_data,
            evaluation_split=m.evaluation_split,
            created_at=m.created_at
        ))
    return results

@router.get("/reports/{analysis_id}/pdf")
def download_pdf_report(analysis_id: int, db: Session = Depends(get_db)):
    """
    Download clinical analysis summary as a PDF report.
    """
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis record not found.")

    pred_res = record.prediction_results or {}
    pdf_path = pdf_service.generate_report(
        analysis_id=record.id,
        model_name=record.model.model_name if record.model else "Unknown",
        architecture=record.model.architecture if record.model else "DenseNet121",
        dataset_name=record.model.dataset_name if record.model else "Chest X-ray",
        overall_assessment=pred_res.get("overall_assessment", "Screening completed"),
        predictions=pred_res.get("predictions", []),
        image_path=record.image_path,
        gradcam_path=record.gradcam_path,
        created_at=record.created_at
    )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"medvision_report_{analysis_id}.pdf"
    )
