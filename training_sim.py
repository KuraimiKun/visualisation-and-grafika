"""Lightweight training loops for interactive visualisations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

Array = np.ndarray


def generate_regression_data(
    seed: int = 0,
    n_samples: int = 200,
    test_size: float = 0.3,
) -> Tuple[Array, Array, Array, Array]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-3.0, 3.0, n_samples).reshape(-1, 1)
    noise = rng.normal(loc=0.0, scale=0.7, size=(n_samples, 1))
    y = 2.2 * x + 0.5 + noise
    return train_test_split(x, y, test_size=test_size, random_state=seed)


def generate_classification_data(
    seed: int = 0,
    n_samples: int = 320,
    test_size: float = 0.3,
) -> Tuple[Array, Array, Array, Array]:
    features, targets = make_moons(n_samples=n_samples, noise=0.25, random_state=seed)
    targets = targets.reshape(-1, 1)
    return train_test_split(features, targets, test_size=test_size, random_state=seed)


class BaseModel:
    """Shared interface for the simulated learners."""

    task_type: str = "generic"

    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.epoch = 0
        self.reset()

    def reset(self) -> None:
        self.epoch = 0

    def set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def predict(self, features: Array) -> Array:
        raise NotImplementedError

    def train_epoch(
        self,
        x_train: Array,
        y_train: Array,
        x_val: Array,
        y_val: Array,
    ) -> Dict[str, Optional[float]]:
        raise NotImplementedError


class LinearRegressionModel(BaseModel):
    task_type = "regression"

    def __init__(self, learning_rate: float) -> None:
        self.input_dim = 1
        super().__init__(learning_rate)

    def reset(self) -> None:
        super().reset()
        self.weights = np.random.randn(self.input_dim, 1) * 0.1
        self.bias = 0.0

    def predict(self, features: Array) -> Array:
        return features @ self.weights + self.bias

    def train_epoch(
        self,
        x_train: Array,
        y_train: Array,
        x_val: Array,
        y_val: Array,
    ) -> Dict[str, Optional[float]]:
        predictions = self.predict(x_train)
        residuals = predictions - y_train
        loss = float(np.mean(residuals ** 2))
        gradient_w = (2.0 / x_train.shape[0]) * (x_train.T @ residuals)
        gradient_b = float(2.0 * np.mean(residuals))
        self.weights -= self.learning_rate * gradient_w
        self.bias -= self.learning_rate * gradient_b
        val_loss = float(np.mean((self.predict(x_val) - y_val) ** 2))
        self.epoch += 1
        return {
            "epoch": self.epoch,
            "train_loss": loss,
            "val_loss": val_loss,
            "train_acc": None,
            "val_acc": None,
        }


class LogisticRegressionModel(BaseModel):
    task_type = "classification"

    def __init__(self, learning_rate: float, input_dim: int = 2) -> None:
        self.input_dim = input_dim
        super().__init__(learning_rate)

    def reset(self) -> None:
        super().reset()
        self.weights = np.random.randn(self.input_dim, 1) * 0.1
        self.bias = 0.0

    @staticmethod
    def _sigmoid(z: Array) -> Array:
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, features: Array) -> Array:
        logits = features @ self.weights + self.bias
        return self._sigmoid(logits)

    def train_epoch(
        self,
        x_train: Array,
        y_train: Array,
        x_val: Array,
        y_val: Array,
    ) -> Dict[str, Optional[float]]:
        probabilities = np.clip(self.predict(x_train), 1e-7, 1.0 - 1e-7)
        loss = float(
            -np.mean(
                y_train * np.log(probabilities)
                + (1.0 - y_train) * np.log(1.0 - probabilities)
            )
        )
        errors = probabilities - y_train
        gradient_w = (x_train.T @ errors) / x_train.shape[0]
        gradient_b = float(np.mean(errors))
        self.weights -= self.learning_rate * gradient_w
        self.bias -= self.learning_rate * gradient_b
        val_probabilities = np.clip(self.predict(x_val), 1e-7, 1.0 - 1e-7)
        val_loss = float(
            -np.mean(
                y_val * np.log(val_probabilities)
                + (1.0 - y_val) * np.log(1.0 - val_probabilities)
            )
        )
        train_accuracy = float(np.mean((probabilities >= 0.5) == y_train))
        val_accuracy = float(np.mean((val_probabilities >= 0.5) == y_val))
        self.epoch += 1
        return {
            "epoch": self.epoch,
            "train_loss": loss,
            "val_loss": val_loss,
            "train_acc": train_accuracy,
            "val_acc": val_accuracy,
        }


class NeuralNetworkModel(BaseModel):
    task_type = "classification"

    def __init__(self, learning_rate: float, hidden_units: int = 4) -> None:
        self.input_dim = 2
        self.hidden_units = hidden_units
        super().__init__(learning_rate)

    def reset(self) -> None:
        super().reset()
        limit1 = np.sqrt(6.0 / (self.input_dim + self.hidden_units))
        limit2 = np.sqrt(6.0 / (self.hidden_units + 1))
        self.weights_hidden = np.random.uniform(
            -limit1, limit1, size=(self.input_dim, self.hidden_units)
        )
        self.bias_hidden = np.zeros((1, self.hidden_units))
        self.weights_output = np.random.uniform(
            -limit2, limit2, size=(self.hidden_units, 1)
        )
        self.bias_output = np.zeros((1, 1))

    @staticmethod
    def _sigmoid(z: Array) -> Array:
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, features: Array) -> Array:
        hidden_linear = features @ self.weights_hidden + self.bias_hidden
        hidden_activation = np.tanh(hidden_linear)
        output_linear = hidden_activation @ self.weights_output + self.bias_output
        return self._sigmoid(output_linear)

    def _forward(self, features: Array) -> Tuple[Array, Dict[str, Array]]:
        hidden_linear = features @ self.weights_hidden + self.bias_hidden
        hidden_activation = np.tanh(hidden_linear)
        output_linear = hidden_activation @ self.weights_output + self.bias_output
        output_activation = self._sigmoid(output_linear)
        cache = {
            "features": features,
            "hidden_linear": hidden_linear,
            "hidden_activation": hidden_activation,
            "output_linear": output_linear,
            "output_activation": output_activation,
        }
        return output_activation, cache

    def train_epoch(
        self,
        x_train: Array,
        y_train: Array,
        x_val: Array,
        y_val: Array,
    ) -> Dict[str, Optional[float]]:
        outputs, cache = self._forward(x_train)
        probabilities = np.clip(outputs, 1e-7, 1.0 - 1e-7)
        loss = float(
            -np.mean(
                y_train * np.log(probabilities)
                + (1.0 - y_train) * np.log(1.0 - probabilities)
            )
        )
        errors = probabilities - y_train
        samples = x_train.shape[0]
        gradient_w_output = cache["hidden_activation"].T @ errors / samples
        gradient_b_output = np.sum(errors, axis=0, keepdims=True) / samples
        hidden_error = errors @ self.weights_output.T
        hidden_delta = hidden_error * (1.0 - np.tanh(cache["hidden_linear"]) ** 2)
        gradient_w_hidden = cache["features"].T @ hidden_delta / samples
        gradient_b_hidden = np.sum(hidden_delta, axis=0, keepdims=True) / samples
        self.weights_output -= self.learning_rate * gradient_w_output
        self.bias_output -= self.learning_rate * gradient_b_output
        self.weights_hidden -= self.learning_rate * gradient_w_hidden
        self.bias_hidden -= self.learning_rate * gradient_b_hidden
        val_outputs = np.clip(self.predict(x_val), 1e-7, 1.0 - 1e-7)
        val_loss = float(
            -np.mean(
                y_val * np.log(val_outputs)
                + (1.0 - y_val) * np.log(1.0 - val_outputs)
            )
        )
        train_accuracy = float(np.mean((probabilities >= 0.5) == y_train))
        val_accuracy = float(np.mean((val_outputs >= 0.5) == y_val))
        self.epoch += 1
        return {
            "epoch": self.epoch,
            "train_loss": loss,
            "val_loss": val_loss,
            "train_acc": train_accuracy,
            "val_acc": val_accuracy,
        }

    def get_weight_matrices(self) -> List[Array]:
        return [self.weights_hidden.copy(), self.weights_output.copy()]


@dataclass
class ModelConfig:
    label: str
    factory: type
    dataset_fn: Callable[..., Tuple[Array, Array, Array, Array]]


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "Linear Regression": ModelConfig(
        label="Linear Regression",
        factory=LinearRegressionModel,
        dataset_fn=generate_regression_data,
    ),
    "Logistic Regression": ModelConfig(
        label="Logistic Regression",
        factory=LogisticRegressionModel,
        dataset_fn=generate_classification_data,
    ),
    "Neural Network": ModelConfig(
        label="Neural Network",
        factory=NeuralNetworkModel,
        dataset_fn=generate_classification_data,
    ),
}


def instantiate_model(model_name: str, learning_rate: float) -> BaseModel:
    config = MODEL_REGISTRY[model_name]
    return config.factory(learning_rate)


def generate_dataset(model_name: str, seed: int) -> Tuple[Array, Array, Array, Array]:
    config = MODEL_REGISTRY[model_name]
    return config.dataset_fn(seed=seed)
