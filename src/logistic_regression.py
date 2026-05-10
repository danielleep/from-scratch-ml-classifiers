import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

class LogisticRegressionGD():
    """
    Logistic Regression classifier trained with batch gradient descent.
    """
    
    def __init__(self, learning_rate=0.0001, max_iter=10000, eps=0.000001, random_state=1):
       
        # Initialize the weights vector with small random values
        self.random_state = random_state
        self.w_ = np.nan
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.class_names_ = None
        self.n_iter_ = 0
        self.loss_history_ = []
            


    def predict_proba(self, X):
        """
        Return predicted probabilities for the positive class.
        """

        if self.w_ is None:
            raise ValueError("Model must be fitted before calling predict_proba.")

        z = X @ self.w_
        return sigmoid(z)
        

    def predict(self, X, threshold=0.5):
        """
        Return the predicted class label according to the threshold.
        """
        probs = self.predict_proba(X)
        y_binary = (probs >= threshold).astype(int)

        return np.where(
            y_binary == 1,
            self.class_names_[1],
            self.class_names_[0]
        )

    
    def bce_loss(self, X, y):
        """
        Calculate binary cross-entropy loss.
        """

        y_01 = np.where(y == self.class_names_[0], 0, 1) # represents the class 0/1 labels

        probs = self.predict_proba(X)

        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)

        loss = -np.mean(y_01 * np.log(probs) + (1 - y_01) * np.log(1 - probs))

        return loss
    

    def fit(self, X, y):
        """
        Fit training data by minimizing BCE loss using batch gradient descent.
        """

        # Store the class labels and convert them to 0/1
        self.class_names_ = np.unique(y)
        y_01 = np.where(y == self.class_names_[0], 0, 1)

        # Initialize weights
        rng = np.random.default_rng(self.random_state)
        self.w_ = 1e-6 * rng.normal(size=X.shape[1])

        # Store loss values for plotting/debugging
        self.loss_history_ = []

        for i in range(self.max_iter):
            # Forward pass: predicted probabilities
            probs = self.predict_proba(X)

            # Numerical stability for log
            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)

            # BCE loss
            curr_loss = -np.mean(
                y_01 * np.log(probs) + (1 - y_01) * np.log(1 - probs)
            )

            self.loss_history_.append(curr_loss)

            # Stop if loss improvement is very small
            if i > 0 and abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.eps:
                break

            # Gradient of BCE
            grad = (1 / X.shape[0]) * X.T @ (probs - y_01)

            # Gradient descent update
            self.w_ -= self.learning_rate * grad

        # Save number of iterations actually used
        self.n_iter_ = i + 1

        return self
