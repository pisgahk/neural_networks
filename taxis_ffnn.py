"""A small feedforward neural network for taxi-fare prediction.

The network uses trip distance as its input and fare amount as its target.
It is implemented from scratch with NumPy: no PyTorch or other ML library.
"""

import csv
from pathlib import Path

import numpy as np

DATASET = Path("datasets/yellow_tripdata_2024-01.csv")


def load_data(path):
    distances = []
    fares = []

    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            try:
                distance = float(row["trip_distance"])
                fare = float(row["fare_amount"])
            except (KeyError, TypeError, ValueError):
                continue

            if np.isfinite(distance) and np.isfinite(fare):
                if distance > 0 and fare > 0:
                    distances.append(distance)
                    fares.append(fare)

    if len(distances) < 10:
        raise ValueError("The dataset does not contain enough valid rows.")

    return np.array(distances, dtype=np.float32).reshape(-1, 1), np.array(
        fares, dtype=np.float32
    ).reshape(-1, 1)


def normalize(values, mean=None, std=None):
    if mean is None:
        mean = values.mean(axis=0)
    if std is None:
        std = values.std(axis=0)
    std = np.maximum(std, 1e-8)
    return (values - mean) / std, mean, std


class FareNetwork:
    """One hidden-layer feedforward neural network."""

    def __init__(self, input_size=1, hidden_size=16, seed=1):
        rng = np.random.default_rng(seed)
        self.weights_1 = rng.normal(0, 0.5, (input_size, hidden_size))
        self.bias_1 = np.zeros((1, hidden_size))
        self.weights_2 = rng.normal(0, 0.5, (hidden_size, 1))
        self.bias_2 = np.zeros((1, 1))

    def predict(self, inputs):
        hidden = np.maximum(0, inputs @ self.weights_1 + self.bias_1)
        return hidden @ self.weights_2 + self.bias_2

    def train(self, inputs, targets, epochs=3000, learning_rate=0.01):
        samples = len(inputs)

        for epoch in range(epochs):
            hidden_values = inputs @ self.weights_1 + self.bias_1
            hidden = np.maximum(0, hidden_values)
            predictions = hidden @ self.weights_2 + self.bias_2

            error = predictions - targets
            loss = np.mean(error**2)

            # Backpropagation for mean squared error.
            output_gradient = 2 * error / samples
            weights_2_gradient = hidden.T @ output_gradient
            bias_2_gradient = output_gradient.sum(axis=0, keepdims=True)
            hidden_gradient = output_gradient @ self.weights_2.T
            hidden_gradient[hidden_values <= 0] = 0
            weights_1_gradient = inputs.T @ hidden_gradient
            bias_1_gradient = hidden_gradient.sum(axis=0, keepdims=True)

            self.weights_2 -= learning_rate * weights_2_gradient
            self.bias_2 -= learning_rate * bias_2_gradient
            self.weights_1 -= learning_rate * weights_1_gradient
            self.bias_1 -= learning_rate * bias_1_gradient

            if epoch % 500 == 0 or epoch == epochs - 1:
                print(f"epoch {epoch:4d} | training MSE: {loss:.4f}")


def main():
    distances, fares = load_data(DATASET)

    split = int(len(distances) * 0.8)
    train_distances, test_distances = distances[:split], distances[split:]
    train_fares, test_fares = fares[:split], fares[split:]

    train_x, distance_mean, distance_std = normalize(train_distances)
    test_x, _, _ = normalize(test_distances, distance_mean, distance_std)
    train_y, fare_mean, fare_std = normalize(train_fares)
    test_y, _, _ = normalize(test_fares, fare_mean, fare_std)

    model = FareNetwork()
    model.train(train_x, train_y)

    normalized_predictions = model.predict(test_x)
    predictions = normalized_predictions * fare_std + fare_mean
    actual_fares = test_y * fare_std + fare_mean
    rmse = np.sqrt(np.mean((predictions - actual_fares) ** 2))

    print(f"\nvalid rows: {len(distances)}")
    print(f"test RMSE: ${rmse:.2f}")
    print("\nSample predictions:")
    for distance, actual, predicted in zip(
        test_distances[:5], actual_fares[:5], predictions[:5]
    ):
        print(
            f"{distance[0]:.2f} miles | actual ${actual[0]:.2f} "
            f"| predicted ${predicted[0]:.2f}"
        )


if __name__ == "__main__":
    main()
