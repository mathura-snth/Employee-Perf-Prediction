import numpy as np


def softmax(Z):
    """Row-wise softmax"""
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def loss_and_grad(W, b, X, y_idx, sample_weight, l2):
    """Weighted cross-entropy + (l2/2)||W||^2 and its gradient."""
    n = len(X)
    P = softmax(X @ W + b)
    rows = np.arange(n)
    nll = -np.log(np.clip(P[rows, y_idx], 1e-15, None))
    loss = np.mean(sample_weight * nll) + 0.5 * l2 * np.sum(W ** 2)

    dZ = P
    dZ[rows, y_idx] -= 1.0
    dZ *= (sample_weight / n)[:, None]
    return loss, X.T @ dZ + l2 * W, dZ.sum(axis=0)


class SoftmaxRegression:
    """Multinomial logistic regression trained by full-batch gradient descent. """

    def __init__(self, learning_rate=0.5, n_iter=5000, l2=1e-3, class_weight=None):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.l2 = l2
        self.class_weight = class_weight

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n, p = X.shape
        K = len(self.classes_)

        if self.class_weight == "balanced":
            class_w = n / (K * np.bincount(y_idx, minlength=K))
            sample_weight = class_w[y_idx]
        elif self.class_weight is None:
            sample_weight = np.ones(n)
        else:
            raise ValueError("class_weight must be None or 'balanced'")

        self.W_, self.b_ = np.zeros((p, K)), np.zeros(K)
        self.loss_history_ = []
        for _ in range(self.n_iter):
            loss, dW, db = loss_and_grad(self.W_, self.b_, X, y_idx, sample_weight, self.l2)
            self.W_ -= self.learning_rate * dW
            self.b_ -= self.learning_rate * db
            self.loss_history_.append(loss)
        return self

    def predict_proba(self, X):
        return softmax(np.asarray(X, dtype=float) @ self.W_ + self.b_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
