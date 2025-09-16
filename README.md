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


## What each assignment implements

### Assignment 1 — Digit classification & k‑NN experiments
- Dataset: MNIST / digits dataset variants.
- Implementations:
  - **Brute-force k‑NN** for baseline comparisons.
  - **KDTree-based FastKNN** (custom implementation) to accelerate neighbor search.
  - **MLPClassifier (scikit‑learn)** experiments with different architectures and hyperparameters.
  - Data preprocessing: normalization, train/test splits, basic feature scaling.
  - Model evaluation: confusion matrices, per-class accuracy, common misclassifications analysis.
  - Visualizations: sample images, misclassified examples, learning curves.

### Assignment 2 — Dimensionality reduction & visualization
Dataset(s): housing-boston.csv; digits dataset (referenced in notebook).
- Implementations:
 - Linear & polynomial regression implementations and mandatory validation tests (Lecture 2).
 - Multivariate regression testing / model validation routines (Lecture 2).
 - Logistic regression: implementations for LogisticRegressionModel and LinearLogisticRegressionModel and related exercises (Lecture 3).
 - Model selection & regularization: ROCAnalysis workflows and ForwardSelection feature-selection experiments (Lecture 4).
 - Neural networks: lecture examples and hands-on exercises (Lecture 5).
 - Data & utilities:
     - Loading and using housing-boston.csv and digits (notebook cells show dataset reading).
     - Uses helper modules/classes referenced in the notebook such as DecisionBoundary, ForwardSelection, MachineLearningModel, ROCAnalysis.
 - Visualizations & checks:
     - Decision boundary / classifier visualization helpers appear in the notebook import list and are used alongside the exercises.

### Assignment 3 — SVMs, clustering & dimensionality reduction

Datasets: datasets loaded in notebook cells (via pandas / sklearn).
 - Implementations:
   - Support Vector Machines (Linear and RBF kernels): data exploration & preprocessing, training, and mandatory tasks to evaluate SVM models (Lecture 6).
   - Hyperparameter tuning tasks for SVMs (grid-style tuning and parameter-sensitivity exercises).
   - Comparative experiments: SVM vs Logistic Regression (non-mandatory comparative tasks).
   - Clustering: mandatory clustering exercises and comparisons (Lecture 8).
   - Dimensionality reduction: PCA, MDS, and t-SNE comparison tasks (Lecture 9), including DR + clustering comparisons.
 - Data & utilities:
   - Common ML stack imports visible in notebook: numpy, pandas, matplotlib, seaborn, scikit-learn.
 - Visualizations & checks:
   - Exploratory data analysis and visualization cells precede modeling tasks; notebooks include plotting and comparisons for model selection and DR/clustering results.
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
