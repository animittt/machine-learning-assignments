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
- `code-duplicate-detection/` — TF–IDF vectorization, Random Forest classifier, thresholding & evaluation for code-similarity.
- `ensemble-retrieval/` — experiments combining multiple similarity metrics (Jaccard, Cosine, TF–IDF, BM25) and retrieval ensembles.
- `presentations/` — slides and notes (e.g., presentation derived from "Visualizing and Understanding Recurrent Networks").
- `utils/` — helper scripts for plotting, metrics, and dataset loaders.

> Note: folder names above reflect the repository structure and may include minor variations.

---

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

### Code duplicate detection & similarity projects
- Implementations / techniques:
  - **TF–IDF vectorization** of code (token/line-based features).
  - **Similarity metrics** experiments (cosine similarity, Jaccard over token sets).
  - **Random Forest** classifier trained to predict whether two files solve the same problem.
  - Threshold tuning and evaluation metrics for retrieval/duplicate detection.
  - Analysis and discussion of false positives / candidate pairs.

### Ensemble retrieval / hybrid similarity method
- Techniques combined:
  - **BM25** and TF–IDF retrieval scores.
  - Lexical set-similarity (Jaccard) and vector-similarity (Cosine).
  - Score fusion / aggregation strategies to improve ranking robustness.
  - Small-scale experiments showing how ensembles affect precision/recall at top‑k.

### Additional experiments & utilities
- Clustering explorations (k‑means / hierarchical) used for exploratory grouping.
- Scripts for building reproducible plots, saving model checkpoints/pickle files, and evaluation dashboards.

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

4. Run cells sequentially. Most notebooks assume the data files are present inside the corresponding assignment folder; follow the top-of-notebook notes if any preprocessing steps are required.

---

## Recommended starting point
- `Assignment-1-2025/` — start with the k‑NN vs MLP notebook to see both algorithmic implementation and standard scikit-learn usage.
- `Assignment-2-2025/` — open the dimensionality reduction comparison notebook to reproduce the 3×3 visualization grid.

---

## Notes & suggested polish for public release
- Add a `requirements.txt` capturing exact versions used for reproducibility.
- Add short one-line summaries inside each assignment folder README so visitors can quickly find experiments of interest.
- Consider adding a small `data/README.md` explaining any large datasets (and whether they are included or need to be downloaded).

---

## License
You can add a license (e.g., MIT) if you want to make the code explicitly reusable.

---

If you want, I can now:
- generate a `requirements.txt` based on the notebooks (best-effort), or
- produce concise one-line summaries for each notebook inside every assignment folder, or
- convert this README into a `README.md` file and open a draft PR-ready version.

Tell me which option you'd like next and I'll continue.

