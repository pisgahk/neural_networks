# Taxi Fare Prediction

## Current `taxis.py` implementation

The `taxis.py` file is a PyTorch learning exercise that estimates a taxi fare from
the trip distance. Its active model is the linear equation:

```text
fare = weight * miles + bias
```

The script contains three approaches:

1. `saada_implementation` manually calculates the mean squared error and its
   gradients, then updates the weight and bias with gradient descent.
2. `lowest_possible_loss` calculates the closed-form least-squares solution for
   the best possible one-feature linear model.
3. `autograd_implementation` uses PyTorch automatic differentiation and the SGD
   optimizer.

It also contains a partially implemented `FareModel` class using `nn.Linear`,
along with model-saving and model-loading helpers. These helpers are not currently
usable as written because `FareModel` is defined inside another function while
`save_model` and `load_model` refer to it outside that scope.

The current `main` function uses a small hard-coded tensor of distances and fares,
starts with a weight of `1.0` and a bias of `0.0`, and runs 10,000 manual gradient
descent iterations. It prints the parameters and loss at every iteration.

## Converting it into a neural-network project

The project becomes a feedforward neural network when the model has one or more
hidden layers and nonlinear activation functions. A suitable initial architecture
is:

```text
input features -> Linear -> ReLU -> Linear -> ReLU -> Linear -> predicted fare
```

For example:

```python
class FareModel(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, features):
        return self.network(features)
```

The model should be generic over the number of input features so that the same
training code can first use distance alone and later include additional features.

## Required additions and implementation work

### 1. Load the prepared dataset

Replace the hard-coded tensors with `datasets/taxi_dataset.csv`. The data-loading
code should:

- Read the CSV with a standard Python CSV or dataframe library.
- Select the feature columns and the `fare_amount` target.
- Report a clear error if the file is missing or required columns are absent.
- Convert the selected values to numeric tensors.

### 2. Clean and validate records

Invalid records should be removed before training. At minimum:

- `trip_distance` must be finite and greater than zero.
- `fare_amount` must be finite and greater than zero.
- Numeric feature values must not be missing or infinite.
- Categorical or identifier fields must be validated before encoding.

The cleaning rules should be applied consistently when preparing training and
inference data.

### 3. Add useful input features

The first neural-network version can use:

- `trip_distance`
- `passenger_count`
- Trip duration calculated from pickup and drop-off timestamps
- Pickup hour
- Day of week
- `PULocationID` and `DOLocationID`, preferably with embeddings or carefully
  managed categorical encoding
- `Airport_fee` or an airport-trip indicator

Do not use `total_amount`, `tip_amount`, `tolls_amount`, tax fields, or other
values that are calculated from the fare. Those fields would leak information
from the target into the input and make evaluation misleading.

### 4. Split the data correctly

Create separate training, validation, and test sets. The test set must remain
unseen until final evaluation. A time-based split is preferable for taxi data:
train on earlier trips and evaluate on later trips. If a random split is used,
make it reproducible with a fixed seed.

### 5. Normalize numeric values

Fit the mean and standard deviation using the training set only:

```text
normalized_value = (value - training_mean) / training_std
```

Reuse those training statistics for validation, testing, and inference. Save the
normalization values with the trained model or in a companion metadata file.

### 6. Implement a real training pipeline

The training code should:

- Use a `TensorDataset` and `DataLoader` for mini-batch training.
- Instantiate the neural-network model with the correct feature count.
- Use an optimizer such as Adam.
- Use `MSELoss` or `HuberLoss`.
- Clear gradients on every iteration.
- Train for a configurable number of epochs.
- Record training and validation loss.
- Use early stopping or retain the best validation checkpoint when appropriate.

The current manual implementation should remain as a simple educational
comparison, but it should not be the main production-style training path.

### 7. Evaluate in fare units

Report at least:

- Mean absolute error (MAE)
- Root mean squared error (RMSE)

If the target is normalized during training, convert predictions back to dollar
amounts before reporting the final metrics. Also compare the neural network with
the current one-feature linear regression baseline.

### 8. Fix model persistence

Move `FareModel` to module scope and update saving/loading so that the model can
be reconstructed reliably. Save:

- `model.state_dict()`
- The feature-column order
- Normalization means and standard deviations
- Model configuration, such as hidden-layer sizes
- A model version

The inference code must apply exactly the same feature engineering and
normalization used during training.

### 9. Add predictions and visual diagnostics

Add an inference path that accepts a new trip and returns a predicted fare. Useful
diagnostics include:

- Actual versus predicted fare
- Residuals versus trip distance
- Training and validation loss curves
- Error distributions

These can reveal systematic errors, such as poor predictions for very long trips
or unusual pickup locations.

### 10. Add focused tests

Tests should cover:

- Invalid-row filtering
- Feature extraction from timestamps
- Consistent feature-column ordering
- Normalization and inverse normalization
- Dataset split reproducibility
- Model output shape
- A short training run that reduces loss
- Saving and loading a model with equivalent predictions

## Suggested implementation order

1. Load `datasets/taxi_dataset.csv` and validate its schema.
2. Build the distance-only neural-network baseline.
3. Add train/validation/test splitting and normalization.
4. Add mini-batch training, metrics, and checkpointing.
5. Add passenger count, duration, and time-based features.
6. Add categorical location handling.
7. Add inference, model metadata, plots, and tests.

## Appropriate project description

Once the hidden layers, nonlinear activations, data pipeline, and evaluation
process are implemented, the project can be described as:

> **Taxi Fare Prediction Using a Feedforward Neural Network**

The current file is best described as a linear-regression and gradient-descent
tutorial. Adding more data or cleaning the CSV improves the experiment, but the
model itself qualifies as a neural network only after the hidden layers and
nonlinear activations are implemented and trained.
