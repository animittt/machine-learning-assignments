from abc import ABC, abstractmethod
import numpy as np

class MachineLearningModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        pass

def _polynomial_features(X, degree):
    """
        Generate polynomial features from the input features.
        Check the slides for hints on how to implement this one. 
        This method is used by the regression models and must work
        for any degree polynomial
        Parameters:
        X (array-like): Features of the data.

        Returns:
        X_poly (array-like): Polynomial features.
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape
    X_poly = np.ones((n_samples, 1))

    for d in range(1, degree + 1):
        for i in range(n_features):
            X_poly = np.hstack((X_poly, (X[:, [i]] ** d)))

    return X_poly
    

def map_feature_interactions(X1, X2, degree):
    """
    Nonlinear polynomial feature expansion for two features (with interactions).
    """
    m = X1.shape[0]
    out = np.ones((m, 1))

    for i in range(1, degree + 1):
        for j in range(i + 1):
            term = (X1 ** (i - j)) * (X2 ** j)
            out = np.hstack((out, term.reshape(-1, 1)))

    return out

class RegressionModelNormalEquation(MachineLearningModel):
    """
    Class for regression models using the Normal Equation for polynomial regression.
    """

    def __init__(self, degree):
        """
        Initialize the model with the specified polynomial degree.

        Parameters:
        degree (int): Degree of the polynomial features.
        """
        self.degree = degree
        self.theta = None

    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        X_poly = _polynomial_features(X, self.degree)
        y = np.asarray(y).reshape(-1, 1)
        # Normal Equation: theta = (X^T * X)^(-1) * X^T * y
        self.theta = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        X_poly = _polynomial_features(X, self.degree)
        predictions = X_poly.dot(self.theta)
        return predictions
    

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        y = np.asarray(y).reshape(-1, 1)
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return mse

class RegressionModelGradientDescent(MachineLearningModel):
    """
    Class for regression models using gradient descent optimization.
    """

    def __init__(self, degree, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the model with the specified parameters.

        Parameters:
        degree (int): Degree of the polynomial features.
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for gradient descent.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        X_poly = _polynomial_features(X, self.degree)
        y = np.asarray(y).reshape(-1, 1)
        n_samples, n_features = X_poly.shape

        self.theta = np.zeros((n_features, 1))

        for _ in range(self.num_iterations):
            predictions = X_poly @ self.theta
            gradients = (2 / n_samples) * X_poly.T @ (predictions - y)
            self.theta -= self.learning_rate * gradients
            cost = np.mean((predictions - y) ** 2)
            self.cost_history.append(cost)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        X_poly = _polynomial_features(X, self.degree)
        return X_poly @ self.theta

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        y = np.asarray(y).reshape(-1, 1)
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return mse

class LogisticRegression(MachineLearningModel):
    """
    Logistic Regression model using gradient descent optimization.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the logistic regression model.

        Parameters:
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        m, n = X.shape

        X_ext = np.hstack([np.ones((m, 1)), X])
        self.theta = np.zeros((n + 1, 1))
        for _ in range(self.num_iterations):
            h = self._sigmoid(X_ext @ self.theta)
            gradient = (1/m) * X_ext.T @ (h - y)
            self.theta -= self.learning_rate * gradient
            cost = self._cost_function(X_ext, y)
            self.cost_history.append(cost)

    def predict(self, X):
        """
        Make predictions using the trained logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        X = np.asarray(X)
        m = X.shape[0]
        X_ext = np.hstack([np.ones((m, 1)), X])
        predictions = self._sigmoid(X_ext @ self.theta)
        return predictions

    def evaluate(self, X, y):
        """
        Evaluate the logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (e.g., accuracy).
        """
        probs = self.predict(X)
        preds = (probs >= 0.5).astype(int)
        accuracy = np.mean(preds.flatten() == y.flatten())
        return accuracy

    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        z = np.asarray(z)
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        m = len(y)
        h = self._sigmoid(X @ self.theta)
        epsilon = 1e-15
        cost = -(1/m) * (y.T @ np.log(h + epsilon) + (1 - y).T @ np.log(1 - h + epsilon))
        return cost.item() 

class NonLinearLogisticRegression(MachineLearningModel):
    """
    Nonlinear Logistic Regression model using gradient descent optimization.
    Works for two features with polynomial feature mapping.
    """

    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the nonlinear logistic regression model.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.cost_history = []

    def _sigmoid(self, z):
        """
        Sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.
        """
        m = X.shape[0]
        h = self._sigmoid(X @ self.theta)
        epsilon = 1e-15
        cost = -(1/m) * (y.T @ np.log(h + epsilon) + (1 - y).T @ np.log(1 - h + epsilon))
        return cost.item()

    def fit(self, X, y):
        """
        Train the nonlinear logistic regression model using gradient descent.
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        if X.shape[1] != 2:
            raise ValueError("Input must have exactly two features.")

        X_mapped = map_feature_interactions(X[:,0], X[:,1], self.degree)

        m, n = X_mapped.shape
        self.theta = np.zeros((n, 1))

        for _ in range(self.num_iterations):
            h = self._sigmoid(X_mapped @ self.theta)
            gradient = (1/m) * X_mapped.T @ (h - y)
            self.theta -= self.learning_rate * gradient
            cost = self._cost_function(X_mapped, y)
            self.cost_history.append(cost)

    def predict(self, X):
        """
        Make predictions using the trained nonlinear logistic regression model.
        """
        X = np.asarray(X)

        if X.shape[1] != 2:
            raise ValueError("Input must have exactly two features.")

        X_mapped = map_feature_interactions(X[:, 0], X[:, 1], self.degree)
        probs = self._sigmoid(X_mapped @ self.theta)
        return probs

    def evaluate(self, X, y):
        """
        Evaluate the model using accuracy.
        """
        probs = self.predict(X)
        preds = (probs >= 0.5).astype(int)
        accuracy = np.mean(preds.flatten() == y.flatten())
        return accuracy
