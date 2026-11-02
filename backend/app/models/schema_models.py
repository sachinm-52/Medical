import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from backend.app.database.session import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    analyses = relationship("AnalysisRecord", back_populates="user")


class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(255), unique=True, nullable=False)
    architecture = Column(String(100), nullable=False)  # DenseNet121, ResNet50, EfficientNet-B0
    dataset_name = Column(String(100), nullable=False)  # Chest X-ray Pneumonia, NIH ChestX-ray14, CheXpert, MedMNIST (ChestMNIST)
    model_version = Column(String(50), default="1.0.0")
    weights_path = Column(String(500), nullable=True)
    status = Column(String(50), default="ready")  # ready, training, not_trained
    task_type = Column(String(50), default="multi_label") # binary, multi_label
    num_classes = Column(Integer, default=14)
    class_labels = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    analyses = relationship("AnalysisRecord", back_populates="model")
    metrics = relationship("ModelMetric", back_populates="model", cascade="all, delete-orphan")


class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    image_filename = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
    gradcam_path = Column(String(500), nullable=True)
    prediction_results = Column(JSON, nullable=False)
    processing_time = Column(Float, nullable=False)  # seconds
    status = Column(String(50), default="completed")
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="analyses")
    model = relationship("ModelRecord", back_populates="analyses")


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    metric_name = Column(String(100), nullable=False)  # accuracy, roc_auc, pr_auc, precision, recall, f1, per_class_roc_auc, confusion_matrix
    metric_value = Column(Float, nullable=True)
    metric_data = Column(JSON, nullable=True)  # for per-class metrics, confusion matrices, loss curves
    evaluation_split = Column(String(50), default="test")  # train, val, test
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    model = relationship("ModelRecord", back_populates="metrics")
