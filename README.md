# Machine Learning Assignments

A curated collection of my machine-learning coursework and project notebooks (2024–2025). Each assignment folder is self-contained and focuses on hands-on implementation: data cleaning and preprocessing, exploratory data analysis (EDA), model building, evaluation, and visualization.

---

## Quick summary

This repository gathers practical notebooks and small scripts used during a machine-learning course and related projects. The notebooks are written in Python (Jupyter) and cover classical supervised learning (classification & regression), dimensionality reduction and visualization, clustering, and retrieval/similarity techniques. Wherever relevant, notebooks show data-loading, feature engineering, model training, hyperparameter tuning, evaluation and plots that reproduce the experiments step-by-step.

---

## Contents / Table of contents

- `Assignment-1-2025/` — classification-focused notebooks (k-NN, KDTree, MLP) and MNIST experiments.
- `Assignment-2-2025/` — dimensionality reduction & visualization (PCA, MDS, t-SNE) across multiple datasets.
- `Assignment-3-2025/` — regression analysis (mtcars), residual diagnostics, and model comparison.
- 


## What each assignment implements (topic‑level detail)

### Assignment 1 — Digit classification & k‑NN experiments
- Dataset(s): MNIST / digits dataset variants.
- Implementations:
  - **Brute-force k‑NN** for baseline comparisons.
  - **KDTree-based FastKNN** (custom implementation) to accelerate neighbor search.
  - **MLPClassifier (scikit‑learn)** experiments with different architectures and hyperparameters.
  - Data preprocessing: normalization, train/test splits, basic feature scaling.
  - Model evaluation: confusion matrices, per-class accuracy, common misclassifications analysis.
  - Visualizations: sample images, misclassified examples, learning curves.

### Assignment 2 — Dimensionality reduction & visualization
- Datasets: MNIST Fashion, Letter Recognition (A–Z), Human Activity Recognition (HAR) (561-feature sensor vectors).
- Implementations:
  - **Principal Component Analysis (PCA)** for variance explanation and projection.
  - **Multidimensional Scaling (MDS)** for preserving pairwise distances in 2D/3D.
  - **t‑SNE** for local neighborhood structure and class-cluster visualization.
  - Combined comparisons: 3×3 scatterplot matrix showing PCA/MDS/t‑SNE across three datasets.
  - Short analysis on **class separability** and visualization-driven insights.

### Assignment 3 — Regression analysis & diagnostics
- Dataset: `mtcars` (fuel efficiency modeling)
- Implementations:
  - Exploratory plots and variable selection.
  - **Linear regression** modeling (OLS) and interpretation of coefficients.
  - Residual diagnostics: heteroscedasticity checks, QQ‑plots, influence/leverage points.
  - Prediction example for a median car and discussion of model assumptions.

---

## How to run

1. Install Python 3.8+.
2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

3. Launch Jupyter Lab / Notebook and open the desired notebook:

```bash
jupyter lab
# or
jupyter notebook
```
