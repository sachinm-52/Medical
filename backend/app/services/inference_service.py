"""
Inference Service for MedVision AI.
Preprocesses uploaded chest X-rays, runs deep learning inference with Sigmoid activations,
and produces structured confidence scores without hallucinated diagnostics.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image
import torch
import torchvision.transforms as transforms
from ml.architectures.factory import create_model
from backend.app.schemas.api_schemas import DiseasePrediction

INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_confidence_level(score: float) -> str:
    if score >= 0.75:
        return "High"
    elif score >= 0.50:
        return "Moderate"
    elif score >= 0.25:
        return "Low"
    else:
        return "Very Low"

class InferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models: Dict[str, torch.nn.Module] = {}

    def load_model(
        self,
        architecture: str,
        num_classes: int,
        task_type: str,
        weights_path: str
    ) -> torch.nn.Module:
        cache_key = f"{architecture}_{weights_path}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        model = create_model(
            architecture=architecture,
            num_classes=num_classes,
            task_type=task_type,
            weights_path=weights_path if Path(weights_path).exists() else None,
            device=self.device
        )
        model.eval()
        self.loaded_models[cache_key] = model
        return model

    def predict(
        self,
        image_path: str,
        model: torch.nn.Module,
        class_labels: List[str],
        task_type: str,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executes model inference on a validated chest X-ray image.
        """
        start_time = time.time()
        image = Image.open(image_path).convert("RGB")
        input_tensor = INFERENCE_TRANSFORMS(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()

        processing_time = round((time.time() - start_time) * 1000, 2)  # ms

        predictions: List[DiseasePrediction] = []

        if task_type == "binary":
            # Binary classification: output logit is probability of positive class (e.g. Pneumonia)
            prob_pneumonia = float(probs)
            prob_normal = 1.0 - prob_pneumonia

            predictions.append(DiseasePrediction(
                disease="NORMAL",
                score=round(prob_normal, 4),
                threshold=threshold,
                is_positive=prob_normal >= threshold,
                confidence_level=get_confidence_level(prob_normal)
            ))
            predictions.append(DiseasePrediction(
                disease="PNEUMONIA",
                score=round(prob_pneumonia, 4),
                threshold=threshold,
                is_positive=prob_pneumonia >= threshold,
                confidence_level=get_confidence_level(prob_pneumonia)
            ))

            top_pred = predictions[1] if prob_pneumonia >= threshold else predictions[0]
            assessment = "Abnormal - Indication of Pneumonia" if prob_pneumonia >= threshold else "Normal - No Pneumonia Detected"

        else:  # Multi-label
            probs_list = probs.tolist() if hasattr(probs, "tolist") else [float(probs)]
            positive_findings = []

            for idx, label in enumerate(class_labels):
                score = float(probs_list[idx]) if idx < len(probs_list) else 0.0
                is_pos = score >= threshold
                pred_item = DiseasePrediction(
                    disease=label,
                    score=round(score, 4),
                    threshold=threshold,
                    is_positive=is_pos,
                    confidence_level=get_confidence_level(score)
                )
                predictions.append(pred_item)
                if is_pos:
                    positive_findings.append(pred_item)

            # Sort predictions descending by score
            predictions.sort(key=lambda x: x.score, reverse=True)
            top_pred = predictions[0] if predictions else None

            if len(positive_findings) > 0:
                disease_names = ", ".join([p.disease for p in positive_findings[:3]])
                assessment = f"Abnormal - Potential signs of: {disease_names}"
            else:
                assessment = "Normal - No thoracic pathology above detection threshold"

        return {
            "predictions": predictions,
            "top_prediction": top_pred,
            "overall_assessment": assessment,
            "processing_time_ms": processing_time
        }

inference_service = InferenceService()
