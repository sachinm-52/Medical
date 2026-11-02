# MedVision AI — System Architecture & Deep Learning Technical Reference

MedVision AI is a full-stack, research-grade medical imaging analysis platform designed to perform explainable thoracic pathology screening on frontal chest X-rays.

```
+-------------------------------------------------------------------------------+
|                             CLIENT / PRESENTATION                             |
|       Next.js 14 App Router • TypeScript • Tailwind CSS • Recharts • Lucide   |
+-------------------------------------------------------------------------------+
                                      |
                           REST APIs (JSON / Multipart)
                                      v
+-------------------------------------------------------------------------------+
|                             FASTAPI BACKEND SERVICE                           |
|       Uvicorn • Pydantic V2 Schemas • SQLAlchemy ORM • ReportLab PDF Engine   |
+-------------------------------------------------------------------------------+
         |                              |                              |
         v                              v                              v
+------------------+         +--------------------+         +-------------------+
|  DATABASE TIER   |         |  INFERENCE ENGINE  |         | EXPLAINABILITY AI |
| SQLite/Postgres  |         | PyTorch Forward    |         | Grad-CAM Engine   |
| Users & Models   |         | DenseNet / ResNet  |         | Feature Hooks     |
| Analyses Records |         | Sigmoid Activations|         | Saliency Heatmaps |
| Empirical Metrics|         | Confidence Scores  |         | Alpha Blending    |
+------------------+         +--------------------+         +-------------------+
```

---

## 1. Deep Learning Architectures

### Primary Architecture: DenseNet121
- **Rationale**: Dense Convolutional Networks (`features.denseblock1` to `features.denseblock4`) connect each layer to every other layer in a feed-forward fashion. This maximizes feature reuse across shallow radiographic edges and deep pulmonary textures, mitigating gradient vanishing in deep networks.
- **Classification Head**: Replaced standard 1000-class ImageNet head with a custom Dropout (`p=0.2`) and Linear layer outputting $C$ class logits ($C=1$ for binary pneumonia, $C=14$ for multi-label thoracic pathology).
- **Grad-CAM Target Layer**: `features.norm5` (the final Batch Normalization layer preceding global average pooling).

### Comparative Architecture 1: ResNet50
- **Rationale**: Standard residual network with bottleneck residual skip-connections ($F(x) + x$), serving as an established benchmark against DenseNet121.
- **Grad-CAM Target Layer**: `layer4[-1]` (the last bottleneck block).

### Comparative Architecture 2: EfficientNet-B0
- **Rationale**: Compound scaling optimizing network depth, width, and input resolution simultaneously for edge-device efficiency.
- **Grad-CAM Target Layer**: `features[-1]` (the final pointwise convolutional expansion).

---

## 2. Activation & Loss Function Engineering

### Multi-Label Classification
- Chest radiographs commonly feature co-occurring diseases (e.g., Pneumonia and Pleural Effusion). Diseases are non-mutually exclusive.
- **Activation**: Multi-label output logits $z_c$ are transformed using independent element-wise Sigmoids:
  $$\sigma(z_c) = \frac{1}{1 + e^{-z_c}}$$
- **Loss Function**: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss()`):
  $$\mathcal{L} = -\frac{1}{C} \sum_{c=1}^C \left[ y_c \log \sigma(z_c) + (1 - y_c) \log (1 - \sigma(z_c)) \right]$$

### Uncertainty Handling (CheXpert)
- **U-Zeros**: Maps $-1$ uncertain labels to $0.0$ (conservative strategy).
- **U-Ones**: Maps $-1$ uncertain labels to $1.0$ (high sensitivity strategy).
- **U-Ignore**: Employs `MaskedBCEWithLogitsLoss` where uncertain labels have zero gradient contribution.
