from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import datetime

class ModelSummary(BaseModel):
    id: int
    model_name: str
    architecture: str
    dataset_name: str
    model_version: str
    status: str
    task_type: str
    num_classes: int
    class_labels: List[str]
    weights_path: Optional[str] = None
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class DatasetInfo(BaseModel):
    name: str
    slug: str
    description: str
    source_url: str
    task_type: str
    num_classes: int
    classes: List[str]
    provenance: str
    patient_level_split: bool
    total_images_note: str
    license: str
    uncertainty_handling: Optional[str] = None


class DiseasePrediction(BaseModel):
    disease: str
    score: float = Field(..., ge=0.0, le=1.0)
    threshold: float = 0.5
    is_positive: bool
    confidence_level: str  # High, Moderate, Low, Very Low


class PredictionResponse(BaseModel):
    analysis_id: int
    image_filename: str
    image_url: str
    gradcam_url: Optional[str] = None
    model_id: int
    model_name: str
    architecture: str
    dataset_name: str
    task_type: str
    predictions: List[DiseasePrediction]
    top_prediction: Optional[DiseasePrediction] = None
    overall_assessment: str  # Normal, Abnormal / Thoracic Abnormality Indicated
    processing_time_ms: float
    timestamp: datetime.datetime
    disclaimer: str


class GradCamRequest(BaseModel):
    analysis_id: int
    target_class_idx: Optional[int] = None
    alpha: float = 0.5  # blending ratio 0.0 to 1.0


class GradCamResponse(BaseModel):
    analysis_id: int
    original_image_url: str
    gradcam_heatmap_url: str
    gradcam_overlay_url: str
    target_disease: str
    explanation_note: str


class HistoryItem(BaseModel):
    id: int
    model_name: str
    dataset_name: str
    image_filename: str
    image_url: str
    gradcam_url: Optional[str] = None
    predictions: List[DiseasePrediction]
    top_prediction: Optional[DiseasePrediction] = None
    overall_assessment: str
    processing_time: float
    status: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class HistoryListResponse(BaseModel):
    total: int
    items: List[HistoryItem]


class MetricItem(BaseModel):
    id: int
    model_id: int
    model_name: str
    architecture: str
    dataset_name: str
    metric_name: str
    metric_value: Optional[float] = None
    metric_data: Optional[Dict[str, Any]] = None
    evaluation_split: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    status: str
    version: str
    cuda_available: bool
    device: str
    database: str
    timestamp: datetime.datetime
