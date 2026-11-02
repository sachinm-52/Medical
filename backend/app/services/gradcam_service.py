"""
Grad-CAM Service for MedVision AI.
Generates explainability heatmaps and overlays for uploaded chest X-rays.
"""

import uuid
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import torch
import torchvision.transforms as transforms
from ml.explainability.gradcam import GradCAM
from backend.app.config import settings

GRADCAM_STORAGE = settings.STORAGE_DIR / "gradcam"
GRADCAM_STORAGE.mkdir(parents=True, exist_ok=True)

class GradCamService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def generate_gradcam(
        self,
        model: torch.nn.Module,
        image_path: str,
        target_class_idx: Optional[int] = None,
        alpha: float = 0.5
    ) -> Tuple[str, str, str]:
        """
        Runs Grad-CAM and saves the colored heatmap and overlay to storage.
        Returns:
            heatmap_rel_url: static url path for heatmap
            overlay_rel_url: static url path for overlay
            overlay_disk_path: absolute path to overlay on disk
        """
        # Ensure model has Grad-CAM layer
        if not hasattr(model, "get_gradcam_target_layer"):
            raise ValueError(f"Model {type(model)} does not provide get_gradcam_target_layer method.")

        target_layer = model.get_gradcam_target_layer()
        gradcam = GradCAM(model=model, target_layer=target_layer)

        original_pil = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(original_pil).unsqueeze(0).to(self.device)

        try:
            heatmap_raw = gradcam.generate(input_tensor=input_tensor, target_class_idx=target_class_idx)
            heatmap_pil, overlay_pil = gradcam.overlay_heatmap(
                original_image_pil=original_pil,
                heatmap=heatmap_raw,
                alpha=alpha
            )
        finally:
            gradcam.remove_hooks()

        # Save images
        unique_id = uuid.uuid4().hex[:12]
        heatmap_filename = f"heatmap_{unique_id}.png"
        overlay_filename = f"overlay_{unique_id}.png"

        heatmap_save_path = GRADCAM_STORAGE / heatmap_filename
        overlay_save_path = GRADCAM_STORAGE / overlay_filename

        heatmap_pil.save(heatmap_save_path, format="PNG")
        overlay_pil.save(overlay_save_path, format="PNG")

        heatmap_url = f"/api/storage/gradcam/{heatmap_filename}"
        overlay_url = f"/api/storage/gradcam/{overlay_filename}"

        return heatmap_url, overlay_url, str(overlay_save_path)

gradcam_service = GradCamService()
