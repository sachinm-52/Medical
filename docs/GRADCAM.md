# Explainable AI: Grad-CAM Implementation Reference

## Mathematical Formulation

Grad-CAM (Gradient-weighted Class Activation Mapping) computes coarse 2D saliency maps that visualize the discriminative regions activated for a specific thoracic disease class $c$.

### 1. Feature Map Activation & Gradient Extraction
Let $A^k$ represent the $k$-th feature activation map of the target convolutional layer, of spatial height $u$ and width $v$.
Let $y^c$ denote the raw score (pre-sigmoid logit) for class $c$.

The importance weight $\alpha_k^c$ of feature map $k$ is calculated via Global Average Pooling over the spatial gradients:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i,j}^k}$$

where $Z = u \times v$ is the spatial area of the feature map.

### 2. Linear Combination & Rectification
The final saliency localization map $L_{\text{Grad-CAM}}^c$ is computed as a weighted combination of forward activation maps, passed through a ReLU function to prioritize features that have a positive impact on class score:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

### 3. Normalization & Blending
The heat matrix is min-max normalized to $[0, 1]$, upsampled to match the original chest radiograph dimensions $(W, H)$ via bicubic interpolation, colorized with OpenCV's Jet color spectrum, and blended via alpha compositing:

$$I_{\text{blended}} = \alpha \cdot I_{\text{heatmap}} + (1 - \alpha) \cdot I_{\text{original}}$$

---

## Target Layer Hooking Across Architectures

| Architecture | Target Layer Module | PyTorch Accessor |
| :--- | :--- | :--- |
| **DenseNet121** | Final Batch Normalization | `model.features.norm5` |
| **ResNet50** | Final Bottleneck Block | `model.layer4[-1]` |
| **EfficientNet-B0** | Final Feature Conv Block | `model.features[-1]` |
