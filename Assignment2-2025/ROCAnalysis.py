class ROCAnalysis:
    """
    Class to calculate various metrics for Receiver Operating Characteristic (ROC) analysis.

    Attributes:
        y_pred (list): Predicted labels.
        y_true (list): True labels.
        tp (int): Number of true positives.
        tn (int): Number of true negatives.
        fp (int): Number of false positives.
        fn (int): Number of false negatives.
    """

    def __init__(self, y_predicted, y_true):
        """
        Initialize ROCAnalysis object.

        Parameters:
            y_predicted (list): Predicted labels (0 or 1).
            y_true (list): True labels (0 or 1).
        """
        self.y_pred = y_predicted
        self.y_true = y_true
        self.tp = sum((yp == 1 and yt == 1) for yp, yt in zip(self.y_pred, self.y_true))
        self.tn = sum((yp == 0 and yt == 0) for yp, yt in zip(self.y_pred, self.y_true))
        self.fp = sum((yp == 1 and yt == 0) for yp, yt in zip(self.y_pred, self.y_true))
        self.fn = sum((yp == 0 and yt == 1) for yp, yt in zip(self.y_pred, self.y_true))

    def tp_rate(self):
        """
        Calculate True Positive Rate (Sensitivity, Recall).

        Returns:
            float: True Positive Rate.
        """
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0

    def fp_rate(self):
        """
        Calculate False Positive Rate.

        Returns:
            float: False Positive Rate.
        """
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0

    def precision(self):
        """
        Calculate Precision.

        Returns:
            float: Precision.
        """
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
  
    def f_score(self, beta=1):
        """
        Calculate the F-score.

        Parameters:
            beta (float, optional): Weighting factor for precision in the harmonic mean. Defaults to 1.

        Returns:
            float: F-score.
        """
        precision = self.precision()
        recall = self.tp_rate()
        if (precision + recall) == 0:
            return 0
        beta_squared = beta ** 2
        return (1 + beta_squared) * (precision * recall) / ((beta_squared * precision) + recall)
