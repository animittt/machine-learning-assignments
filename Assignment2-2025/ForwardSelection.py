import numpy as np
from ROCAnalysis import ROCAnalysis
import copy

class ForwardSelection:
    """
    A class for performing forward feature selection based on maximizing the F-score of a given model.

    Attributes:
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        model (object): Machine learning model with fit and predict methods.
        selected_features (list): List of selected feature indices.
        best_cost (float): Best F-score achieved during feature selection.
    """

    def __init__(self, X, y, model):
        """
        Initializes the ForwardSelection object.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            model (object): Machine learning model with fit and predict methods.
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.model = model
        self.selected_features = []
        self.best_cost = 0.0
        self.trained_model = None

    def create_split(self, X, y, split_ratio=0.8, seed=42):
        """
        Creates a train-test split of the data.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.

        Returns:
            X_train (array-like): Features for training.
            X_test (array-like): Features for testing.
            y_train (array-like): Target labels for training.
            y_test (array-like): Target labels for testing.
        """
        np.random.seed(seed)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        split_idx = int(len(indices) * split_ratio)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def train_model_with_features(self, features):
        """
        Trains the model using selected features and evaluates it using ROCAnalysis.

        Parameters:
            features (list): List of feature indices.

        Returns:
            float: F-score obtained by evaluating the model.
        """
        X_selected = self.X[:, features]
        X_train, X_test, y_train, y_test = self.create_split(X_selected, self.y)

        model_copy = copy.deepcopy(self.model)
        model_copy.fit(X_train, y_train)
        y_pred = (model_copy.predict(X_test) >= 0.5).astype(int)

        roc = ROCAnalysis(y_pred, y_test)
        return roc.f_score()

    def forward_selection(self):
        """
        Performs forward feature selection based on maximizing the F-score.
        """
        num_features = self.X.shape[1]
        remaining_features = list(range(num_features))

        while remaining_features:
            best_feature = None
            best_score = self.best_cost

            for feature in remaining_features:
                candidate_features = self.selected_features + [feature]
                score = self.train_model_with_features(candidate_features)

                if score > best_score:
                    best_score = score
                    best_feature = feature

            if best_feature is not None:
                self.selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                self.best_cost = best_score
            else:
                break  # No improvement

    def fit(self):
        """
        Fits the model using the selected features.
        """
        if not self.selected_features:
            self.forward_selection()

        X_selected = self.X[:, self.selected_features]
        X_train, _, y_train, _ = self.create_split(X_selected, self.y)
        self.trained_model = copy.deepcopy(self.model)
        self.trained_model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts the target labels for the given test features.

        Parameters:
            X_test (array-like): Test features.

        Returns:
            array-like: Predicted target labels.
        """
        if self.trained_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        X_test_selected = X_test[:, self.selected_features]
        return (self.trained_model.predict(X_test_selected) >= 0.5).astype(int)

    def get_selected_features(self):
        """
        Returns the indices of the selected features.

        Returns:
            list: Indices of selected features.
        """
        return self.selected_features
    
    def get_best_cost(self):
        """
        Returns the best F-score achieved during feature selection.

        Returns:
            float: Best F-score.
        """
        return self.best_cost
    
    def get_trained_model(self):
        """
        Returns the trained model.

        Returns:
            object: Trained model.
        """
        return self.trained_model