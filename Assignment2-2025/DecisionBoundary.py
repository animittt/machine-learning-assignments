import numpy as np
import matplotlib.pyplot as plt

def plotDecisionBoundary(X1, X2, y, model, title="Decision Boundary"):
    """
    Plots the decision boundary for a binary classification model along with the data points.

    Parameters:
        X1 (array-like): Feature values for the first feature (normalized).
        X2 (array-like): Feature values for the second feature (normalized).
        y (array-like): Target labels (0 or 1).
        model (object): Trained binary classification model with a `predict` method.
        title (str): Title for the plot.

    Returns:
        None
    """
    x_min, x_max = X1.min() - 0.5, X1.max() + 0.5
    y_min, y_max = X2.min() - 0.5, X2.max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    probs = model.predict(grid)
    preds = (probs >= 0.5).astype(int).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds, alpha=0.4, cmap=plt.cm.coolwarm)
    plt.scatter(X1, X2, c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel("Feature 1 (Normalized)")
    plt.ylabel("Feature 2 (Normalized)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
