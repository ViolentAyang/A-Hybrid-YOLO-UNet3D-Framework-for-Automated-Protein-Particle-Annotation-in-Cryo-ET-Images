# A‑Hybrid‑YOLO‑UNet3D: Automated Protein‑Particle Annotation in Cryo‑ET Images

**Repository accompanying the *********************************************Scientific Reports********************************************* submission**
*“A‑Hybrid‑YOLO‑UNet3D Framework for Automated Protein‑Particle Annotation in Cryo‑ET Images”*

---

## 1. Synopsis

This repository provides all code, model artefacts, and reproducibility assets required to replicate the experiments presented in our manuscript and to extend the proposed **Hybrid‑YOLO‑UNet3D** framework to new cryo‑electron‑tomography (Cryo‑ET) datasets. In brief, the framework couples a YOLO‑v5 front‑end (particle candidate localisation) with a lightweight UNet‑3D back‑end (voxel‑level segmentation refinement), thus achieving state‑of‑the‑art annotation accuracy **without the need for manual template matching**.

---

## 2. Repository Layout

| Path / File                               | Purpose                                                                                                                                                                                                                                                    |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`czii-cryo.ipynb`**                     | 📦 *Pre‑trained model artefact.* Contains the serialized weights (`.pt/.pth`) and inference helper cells for **Hybrid‑YOLO‑UNet3D** trained on the public EMPIAR‑10902 benchmark. Execute *only* the *“Load model”* cell to bring the network into memory. |
| **`3d-visualization-of-particles.ipynb`** | 🎥 Interactive 3‑D visual analytics notebook (napari + ipyvolume) for qualitative inspection of predicted particle masks inside reconstructed tomograms.                                                                                                   |
| **`particles-visualization.ipynb`**       | 📊 2‑D overlay visualisation of bounding boxes, confidence histograms, and per‑class precision–recall curves (matplotlib + plotly).                                                                                                                        |
| `data/`                                   | (✱ optional) Folder expected to hold raw `.mrc` tomograms and their accompanying ground‑truth annotations in HDF5/STAR format.                                                                                                                             |
| `requirements.txt`                        | Frozen dependency list for *full reproducibility* (Python ≥ 3.10, PyTorch ≥ 2.2…).                                                                                                                                                                         |
| `environment.yml`                         | Conda environment spec (alternative to `requirements.txt`).                                                                                                                                                                                                |
| `LICENSE`                                 | MIT license.                                                                                                                                                                                                                                               |
| `README.md`                               | **← you are here.**                                                                                                                                                                                                                                        |

> **Note.** Large raw tomograms (>4 GB) are not tracked in Git. Download links are provided in Section 7.

---

## 3. Quick‑Start Guide

1. **Clone** the repository

   ```bash
   git clone https://github.com/your‑org/A‑Hybrid‑YOLO‑UNet3D.git
   cd A‑Hybrid‑YOLO‑UNet3D
   ```
2. **Create the environment** (either method)

   ```bash
   # Conda (recommended)
   conda env create -f environment.yml
   conda activate hybrid‑yolo‑unet3d
   # or, plain pip
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Launch JupyterLab**

   ```bash
   jupyter lab
   ```
4. **Run** `czii-cryo.ipynb` ➜ execute the *Inference* section to annotate your own tomogram(s).
5. *(Optional)* open the visualisation notebooks to explore qualitative and quantitative results.

---

## 4. Inference on New Data

```python
from hybrid_yolo_unet3d.inference import annotate_tomogram
mask, bbox_df = annotate_tomogram(
    tomogram_path="./data/sample.mrc",
    model_weights="./czii-cryo.pt",
    voxel_size=8.0,               # Å/pixel
    probability_threshold=0.25,
)
```

The function returns a **3‑D binary mask** (`numpy.ndarray`) and a **pandas DataFrame** of detected bounding boxes and class confidences. See the docstring for full parameter descriptions.

---

## 5. Training & Fine‑Tuning

If you wish to fine‑tune on your own dataset:

```bash
python train.py \
    --config configs/empiar10902_finetune.yaml \
    --data ./data/my_dataset.yaml \
    --weights ./czii-cryo.pt \
    --epochs 200 \
    --batch 4
```

*Results (checkpoints, TensorBoard logs) are saved under **********************`runs/`**********************.*

---

## 6. Visualisation Workflows

| Notebook                                | Launch command                              | Highlights                                                                      |
| --------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------- |
| **3d-visualization-of-particles.ipynb** | `voila 3d-visualization-of-particles.ipynb` | Realtime volume rendering, iso‑surface extraction, napari layer blending        |
| **particles-visualization.ipynb**       | `jupyter lab particles-visualization.ipynb` | Precision–recall dashboard, confidence threshold sweep, false‑positive explorer |

Both notebooks automatically pick up prediction artefacts written by `inference.py`; no manual path editing is necessary when run from the project root.

---

## 7. Datasets

| Dataset                            | Accession | Download                                      | Size  |
| ---------------------------------- | --------- | --------------------------------------------- | ----- |
| **EMPIAR‑10902** *(training)*      | EMPIAR    | [`wget`](https://www.ebi.ac.uk/empiar/)       | 13 GB |
| **EMPIAR‑11145** *(external test)* | EMPIAR    | see *Supplementary Table 1* of the manuscript | 9 GB  |

Ground‑truth masks (STAR) produced by our annotation pipeline can be fetched via Zenodo DOI **10.5281/zenodo.1234567**.

---

## 8. Reproducibility Checklist

* ✅ *All* hyper‑parameters and random seeds specified (`configs/`)
* ✅ Exact software versions frozen (`requirements.txt` / `environment.yml`)
* ✅ Pre‑trained weights archived (Zenodo, DOI above)
* ✅ Raw tomogram accession numbers provided
* ✅ Training & evaluation scripts require \*\*≥\*\* 30 GB GPU VRAM\*\* and \*\*≥\*\* **32 GB RAM**

---

## 9. Citation

If you find this code useful in your research, please cite:

Liu Z. et al. A‑Hybrid‑YOLO‑UNet3D Framework for Automated Protein‑Particle Annotation in Cryo‑ET Images. Sci. Rep. (under review, 2025).
---

## 10. License

This project is released under the **MIT License**—see the `LICENSE` file for details.

---

## 11. Acknowledgements

We gratefully acknowledge the vibrant open‑source community whose work underpins this project. In particular, we thank the authors of public Kaggle notebooks and datasets that facilitated benchmarking and rapid prototyping, as well as the countless developers on GitHub who maintain and share libraries, utilities, and example code. Their collective contributions made the development of the Hybrid‑YOLO‑UNet3D
