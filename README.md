# MedVision AI – Explainable Multi-Disease Detection Using Chest X-rays

[![Next.js](https://img.shields.io/badge/Frontend-Next.js%2014-black)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%202.x-orange)](https://pytorch.org/)
[![Explainable AI](https://img.shields.io/badge/Explainability-Grad--CAM-blue)](https://arxiv.org/abs/1610.02391)
[![License](https://img.shields.io/badge/License-MIT%20%2F%20CC%20BY%204.0-emerald)](LICENSE)

**MedVision AI** is a professional, modern, and interpretable clinical decision-support research platform for detecting multi-label thoracic pathologies from frontal chest radiographs. Built with PyTorch, FastAPI, Next.js, and Grad-CAM, the platform pairs deep learning classification scores with visual activation heatmaps and downloadable PDF radiology reports.

---

## 1. Key Features

- **Multi-Disease Screening**: Simultaneous detection of up to 14 thoracic disease categories (Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia).
- **Explainable AI with Grad-CAM**: Visual saliency heatmaps with dynamic alpha-blending sliders, pathology target switching, side-by-side comparative views, and high-resolution PNG export.
- **Strict Dataset Provenance**: Rigorous data pipelines for the four authorized datasets with zero silent merging and leak-free patient-level cohort splitting.
- **Comparative Backbones**: Benchmarked implementations of **DenseNet121** (primary), **ResNet50**, and **EfficientNet-B0**.
- **Interactive Radiology Dashboard**: Drag-and-drop X-ray uploading, clinical sample picker, adjustable decision thresholds, and database-backed screening history.
- **Clinical PDF Export**: Automated generation of standardized medical analysis summary reports via ReportLab.
- **No Mock Predictions**: All predictions and evaluation curves are generated from genuine model checkpoints and empirical test evaluations.

---

## 2. System Architecture

```
+-------------------------------------------------------------------------------+
|                            MEDVISION AI PLATFORM                              |
+-------------------------------------------------------------------------------+
         |
         +--> FRONTEND: Next.js 14 App Router, TypeScript, Tailwind CSS, Recharts
         |
         +--> REST API: FastAPI, Pydantic V2, SQLAlchemy, ReportLab PDF Engine
         |
         +--> ML PIPELINE: PyTorch 2.x, Torchvision, Scikit-Learn
         |      |
         |      +--> DenseNet121 (Primary Backbone, norm5 Grad-CAM hook)
         |      +--> ResNet50 (Residual Backbone, layer4[-1] hook)
         |      +--> EfficientNet-B0 (Compound Scaling Backbone, features[-1] hook)
         |
         +--> DATASETS (Strict 4-Dataset Provenance)
                +--> Chest X-ray Pneumonia Dataset (Kaggle) - Binary
                +--> NIH ChestX-ray14 - 14 Multi-Label Classes
                +--> Stanford CheXpert - Multi-Label with Uncertainty Handling
                +--> MedMNIST v2 (ChestMNIST Subset) - Lightweight 14 Classes
```

---

## 3. Technology Stack

- **Frontend**: Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS, Lucide React, Recharts.
- **Backend**: Python 3.11, FastAPI, Uvicorn, Pydantic V2, SQLAlchemy, ReportLab.
- **Deep Learning**: PyTorch, Torchvision, Scikit-Learn, MedMNIST.
- **Computer Vision & Saliency**: OpenCV, Pillow, NumPy, Grad-CAM.
- **Database**: SQLite (local development) / PostgreSQL & Supabase (production).

---

## 4. Supported Datasets & Strict Provenance

| Dataset | Source | Task Type | Classes | Patient Split |
| :--- | :--- | :--- | :--- | :--- |
| **Chest X-ray Pneumonia** | [Kaggle Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | Binary | 2 (Normal vs Pneumonia) | Yes (Cohort Split) |
| **NIH ChestX-ray14** | [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) | Multi-label | 14 Thoracic Diseases | Yes (Patient ID) |
| **Stanford CheXpert** | [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/) | Multi-label | 14 Classes (with Uncertainty) | Yes (`patient\d+`) |
| **MedMNIST (ChestMNIST)** | [MedMNIST Zenodo](https://medmnist.com/) | Multi-label | 14 Thoracic Diseases | Yes (Native Split) |

---

## 5. Quickstart & Installation

### Prerequisites
- Node.js >= 18.x
- Python >= 3.10
- Git

### 1. Clone & Set Up Backend Environment

```bash
# Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # On Windows
# source .venv/bin/activate  # On Linux / macOS

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 2. Set Up Frontend Environment

```bash
cd frontend
npm install
cd ..
```

### 3. Initialize Checkpoints & Benchmark Models

Run the automatic data extractor and benchmark trainer:

```bash
python ml/scripts/train_quick_benchmarks.py
python ml/scripts/train_dataset_models.py
```

### 4. Run the Servers

**Terminal 1 — Backend API Server (FastAPI):**
```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be accessible at: `http://localhost:8000/docs`

**Terminal 2 — Frontend Application (Next.js):**
```bash
cd frontend
npm run dev
```
Web application will be accessible at: `http://localhost:3000`

---

## 6. Model Training & Evaluation CLI

MedVision AI provides dedicated training pipelines for all four datasets with customizable architectures, learning rates, epochs, and uncertainty policies.

### Training on MedMNIST ChestMNIST
```bash
python ml/training/train.py --dataset chestmnist --architecture DenseNet121 --epochs 10 --batch_size 32 --lr 1e-4
```

### Training on Kaggle Pneumonia Dataset
```bash
python ml/training/train.py --dataset pneumonia --data_dir ./data/pneumonia --architecture DenseNet121 --epochs 15
```

### Training on NIH ChestX-ray14 (Patient-Level Leakage Prevention)
```bash
python ml/training/train.py --dataset nih --data_dir ./data/nih --architecture ResNet50 --epochs 10
```

### Training on CheXpert with Uncertainty Policy
```bash
# Options: --uncertainty_policy u-zeros, u-ones, or u-ignore
python ml/training/train.py --dataset chexpert --data_dir ./data/chexpert --uncertainty_policy u-zeros --epochs 10
```

---

## 7. Backend API Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Backend operational status and compute device (CPU/CUDA) |
| `GET` | `/api/models` | List available models and weight readiness status |
| `GET` | `/api/datasets` | Retrieve supported dataset metadata and provenance |
| `POST` | `/api/upload` | Upload and validate chest radiograph (JPG/PNG) |
| `POST` | `/api/predict` | Run neural network inference and initial Grad-CAM |
| `POST` | `/api/gradcam` | Generate targeted Grad-CAM heatmap with alpha slider |
| `GET` | `/api/history` | Audit log of previous analyses with search and filters |
| `GET` | `/api/history/{id}` | Detailed findings for an individual analysis |
| `DELETE` | `/api/history/{id}` | Delete analysis record from database |
| `GET` | `/api/reports/{id}/pdf` | Stream generated PDF clinical summary report |
| `GET` | `/api/metrics` | Empirical evaluation metrics across models and splits |

---

## 8. Database Schema

The database model tracks users, trained models, individual analyses, and empirical evaluation metrics:

```sql
users (id, name, email, password_hash, created_at)
models (id, model_name, architecture, dataset_name, model_version, weights_path, status, task_type, num_classes, class_labels, created_at)
analyses (id, user_id, model_id, image_filename, image_path, gradcam_path, prediction_results, processing_time, status, created_at)
model_metrics (id, model_id, metric_name, metric_value, metric_data, evaluation_split, created_at)
```

To initialize or migrate PostgreSQL on Supabase, execute `database/schema.sql`.

---

## 9. Security & Medical Safety Compliance

- **Upload Validation**: File extension validation (`.jpg`, `.jpeg`, `.png`), magic byte inspection, and strict 25MB file size limits.
- **Patient Privacy**: No patient names or sensitive medical identifiers are stored. Images are hashed and stored with anonymized UUID identifiers.
- **Model Score Clarification**: The platform explicitly states that model output probabilities represent algorithmic patterns, not definitive medical diagnoses.
- **Explainability Caveats**: Grad-CAM heatmaps are explicitly presented as feature attribution explanations, not proof of clinical lesion boundaries.

---

## 10. Clinical & Medical Disclaimer

> **IMPORTANT NOTICE:** MedVision AI is developed strictly for **educational, academic, and research purposes**. The predictions and Grad-CAM visualizations generated by this system are algorithmic estimates and are **NOT** intended to serve as confirmed medical diagnoses, diagnostic advice, or treatment guidelines. Clinical chest radiograph interpretation must always be performed by a board-certified radiologist or licensed physician.
