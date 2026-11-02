# Supported Datasets & Strict Provenance Protocols

MedVision AI strictly supports **ONLY** the four medical datasets specified below. No other datasets are imported, combined, or trained on.

---

## Dataset 1: Chest X-ray Pneumonia Dataset
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Task**: Binary Classification (`0: NORMAL`, `1: PNEUMONIA`)
- **Total Images**: 5,863 pediatric chest X-rays (anterior-posterior).
- **Patient Leakage Prevention**: Partitioned into `train/`, `val/`, and `test/` cohorts verified by clinical consensus from two expert physicians.
- **Loader Module**: `ml/datasets/chest_xray_pneumonia.py`

### Directory Structure:
```
data/pneumonia/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Dataset 2: NIH ChestX-ray14
- **Source**: [NIH Clinical Center Box](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **Task**: Multi-label Thoracic Classification (14 Diseases):
  1. Atelectasis
  2. Cardiomegaly
  3. Effusion
  4. Infiltration
  5. Mass
  6. Nodule
  7. Pneumonia
  8. Pneumothorax
  9. Consolidation
  10. Edema
  11. Emphysema
  12. Fibrosis
  13. Pleural Thickening
  14. Hernia
- **Total Images**: 112,120 frontal-view X-ray images from 30,805 unique patients.
- **Patient Leakage Prevention**: Strict patient-level grouping (`Patient ID` column in `Data_Entry_2017.csv`). All images from a single patient are assigned exclusively to either Train, Validation, or Test.
- **Loader Module**: `ml/datasets/nih_chestxray14.py`

---

## Dataset 3: Stanford CheXpert
- **Source**: [Stanford ML Group CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
- **Task**: Multi-label Chest X-ray Classification with Uncertainty Handling.
- **Total Radiographs**: 224,316 chest radiographs from 65,240 patients.
- **Uncertainty Policies**:
  - `u-zeros`: Maps $-1$ uncertain labels to $0.0$.
  - `u-ones`: Maps $-1$ uncertain labels to $1.0$.
  - `u-ignore`: Masked gradient loss calculation.
- **Patient Leakage Prevention**: Extracted via `patient\d+` regex from file paths.
- **Loader Module**: `ml/datasets/chexpert.py`

---

## Dataset 4: MedMNIST v2 (ChestMNIST Subset)
- **Source**: [MedMNIST Benchmark](https://medmnist.com/)
- **Zenodo DOI**: [10.5281/zenodo.6496656](https://zenodo.org/records/6496656)
- **Task**: Lightweight Multi-label Classification (14 classes identical to NIH ChestX-ray14).
- **Split**: Train (78,468), Validation (11,219), Test (22,433).
- **Native Resolution**: 28x28 grayscale patches (expanded to 3 channels and resized to 224x224 for backbone architectures).
- **Patient Leakage Prevention**: Natively split at patient level by dataset authors.
- **Loader Module**: `ml/datasets/chestmnist.py`
