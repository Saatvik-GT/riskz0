import numpy as np
from typing import Optional, Tuple, List, Dict
import sys

sys.path.append(".")
from src.utils.metrics import softmax, one_hot_encode, accuracy_score
from config import NUM_CLASSES


def compute_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Compute inverse-frequency class weights."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(y) / (n_classes * counts)
    return weights


class LogisticRegression:
    def __init__(
        self,
        n_features: int,
        n_classes: int = NUM_CLASSES,
        learning_rate: float = 0.1,
        l2_lambda: float = 0.001,
        batch_size: int = 64,
        epochs: int = 1000,
        early_stopping_patience: int = 50,
        verbose: bool = True,
        momentum: float = 0.0,
        lr_schedule: str = "constant",
        use_class_weights: bool = False,
        grad_clip: float = 0.0,
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.momentum = momentum
        self.lr_schedule = lr_schedule
        self.use_class_weights = use_class_weights
        self.grad_clip = grad_clip

        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None

        self._vW: Optional[np.ndarray] = None
        self._vb: Optional[np.ndarray] = None

        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.train_acc_history: List[float] = []
        self.val_acc_history: List[float] = []

        self._fitted = False

    def _initialize_weights(self):
        np.random.seed(42)
        scale = np.sqrt(2.0 / self.n_features)
        self.weights = np.random.randn(self.n_features, self.n_classes) * scale
        self.bias = np.zeros(self.n_classes)
        self._vW = np.zeros_like(self.weights)
        self._vb = np.zeros_like(self.bias)
        self._fitted = True

    def _get_lr(self, epoch: int) -> float:
        """Get learning rate for the current epoch based on schedule."""
        if self.lr_schedule == "cosine":
            return self.learning_rate * 0.5 * (1 + np.cos(np.pi * epoch / self.epochs))
        elif self.lr_schedule == "step":
            factor = 0.5 ** (epoch // (self.epochs // 4 + 1))
            return self.learning_rate * factor
        else:
            return self.learning_rate

    def _clip_gradients(self, dW: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Clip gradients by norm."""
        if self.grad_clip <= 0:
            return dW, db
        norm = np.sqrt(np.sum(dW ** 2) + np.sum(db ** 2))
        if norm > self.grad_clip:
            scale = self.grad_clip / norm
            dW = dW * scale
            db = db * scale
        return dW, db

    def _cross_entropy_loss(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weights: Optional[np.ndarray] = None
    ) -> float:
        n_samples = len(y_true)

        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        if sample_weights is not None:
            log_likelihood = -np.sum(sample_weights[:, None] * y_true * np.log(y_pred)) / n_samples
        else:
            log_likelihood = -np.sum(y_true * np.log(y_pred)) / n_samples

        l2_penalty = (self.l2_lambda / 2) * np.sum(self.weights**2)

        return log_likelihood + l2_penalty

    def _compute_gradients(
        self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(X)

        d_logits = (y_pred - y_true) / n_samples

        if sample_weights is not None:
            d_logits = d_logits * sample_weights[:, None]

        dW = X.T @ d_logits + self.l2_lambda * self.weights
        db = np.sum(d_logits, axis=0)

        return dW, db

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LogisticRegression":
        X_train = np.array(X_train, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.int64)

        if X_train.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X_train.shape[1]}"
            )

        self._initialize_weights()

        y_train_onehot = one_hot_encode(y_train, self.n_classes)

        class_weights = None
        if self.use_class_weights:
            class_weights = compute_class_weights(y_train, self.n_classes)

        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None
        best_bias = None

        n_samples = len(X_train)

        for epoch in range(self.epochs):
            current_lr = self._get_lr(epoch)

            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_onehot[indices]
            y_indices_shuffled = y_train[indices]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]

                batch_sample_weights = None
                if class_weights is not None:
                    y_batch_indices = y_indices_shuffled[i : i + self.batch_size]
                    batch_sample_weights = class_weights[y_batch_indices]

                logits = X_batch @ self.weights + self.bias
                probs = softmax(logits)

                batch_loss = self._cross_entropy_loss(y_batch, probs, batch_sample_weights)
                epoch_loss += batch_loss
                n_batches += 1

                dW, db = self._compute_gradients(X_batch, y_batch, probs, batch_sample_weights)
                dW, db = self._clip_gradients(dW, db)

                if self.momentum > 0:
                    self._vW = self.momentum * self._vW + current_lr * dW
                    self._vb = self.momentum * self._vb + current_lr * db
                    self.weights -= self._vW
                    self.bias -= self._vb
                else:
                    self.weights -= current_lr * dW
                    self.bias -= current_lr * db

            avg_train_loss = epoch_loss / n_batches
            train_pred = self.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)

            self.train_loss_history.append(avg_train_loss)
            self.train_acc_history.append(train_acc)

            if X_val is not None and y_val is not None:
                val_probs = self.predict_proba(X_val)
                y_val_onehot = one_hot_encode(y_val, self.n_classes)
                val_loss = self._cross_entropy_loss(y_val_onehot, val_probs)
                val_pred = self.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)

                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = self.weights.copy()
                    best_bias = self.bias.copy()
                else:
                    patience_counter += 1

                if self.verbose and (epoch + 1) % 50 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    )

                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    self.weights = best_weights
                    self.bias = best_bias
                    break
            else:
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}"
                    )

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X, dtype=np.float64)
        logits = X @ self.weights + self.bias
        return softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_state(self) -> Dict:
        return {
            "weights": self.weights,
            "bias": self.bias,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "learning_rate": self.learning_rate,
            "l2_lambda": self.l2_lambda,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
            "fitted": self._fitted,
        }

    def set_state(self, state: Dict) -> "LogisticRegression":
        self.weights = state["weights"]
        self.bias = state["bias"]
        self.n_features = state["n_features"]
        self.n_classes = state["n_classes"]
        self.learning_rate = state["learning_rate"]
        self.l2_lambda = state["l2_lambda"]
        self.train_loss_history = state["train_loss_history"]
        self.val_loss_history = state["val_loss_history"]
        self.train_acc_history = state["train_acc_history"]
        self.val_acc_history = state["val_acc_history"]
        self._fitted = state["fitted"]
        return self
