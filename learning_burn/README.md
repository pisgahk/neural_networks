# Learning Burn with Rust: A Coding Tutorial

This is a practical introduction to [Burn](https://burn.dev/) 0.20.1. Work through the sections in order and type the examples yourself. The tutorial starts with tensors, then uses those tensors to build and train a small neural network that predicts a taxi fare from trip distance.

The examples use the same style as `learning_burn/taxis.rs`: explicit shapes, generic backends, `Result`-based errors, and tests for data-processing code.

## 0. Run this project

From the repository root:

```bash
cargo fmt
cargo check --bin taxis
cargo test --test taxi_learning
cargo run --bin taxis
```

The taxi program reads `learning_burn/yellow_tripdata_2024-01.parquet`, selects `trip_distance` and `fare_amount`, filters invalid rows, trains on a short 20,000-row sample, evaluates on validation data, prints predictions, and saves a model to `learning_burn/taxi_model.bin`.

The model file is generated output. It is ignored by Git and can be deleted and regenerated at any time.

---

## 1. Your first Burn program

Create a small Rust file with these imports:

```rust
use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};

type B = NdArray;

fn main() {
    let device = Default::default();
    let tensor = Tensor::<B, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    println!("{tensor}");
}
```

Run it as a Cargo binary. A Burn tensor is written as:

```text
Tensor<Backend, Rank, Kind>
```

- `Backend` says where computation runs. `NdArray` is a portable CPU backend.
- `Rank` is the number of dimensions. A matrix is rank 2; a batch of images might be rank 4.
- `Kind` is usually inferred as floating point, integer, or boolean.

> [!IMPORTANT]
>
> ### Different types of `Backend` in Burn.
>
> 1. `NdArray`: is the easiest backend for learning and CPU-based experiments.
> 2. `Wgpu`: runs through the WebGPU ecosystem and can use GPUs supported by Vulkan, Metal, DirectX 12, or WebGPU-compatible platforms.
> 3. `CPU`: Burn also provides a CubeCL-based CPU backend. `NdArray` delegates operations to the ndarray ecosystem, while the Burn CPU backend uses Burn’s CubeCL execution system.
> 4. `Candle`: is another tensor backend supported, it can support CPU executionand depending on enabled features.
> 5. `LibTorch` / `tch`: Burn can use LibTorch through the `tch` backend.
> 6. `Cuda`: Burn has a CUDA-specific backend. For NVIDIA GPUs.
> 7. `Rocm`: For AMD GPUs.
> 8. `Remote`: Allows tensor operations to be sent to another process or machine. It is more relevant for distributed or service-based systems than for learning tensor fundamentals.
> 9. Router: The router backend can help direct operations between backend implementations. It is useful for advanced runtime and multi-device scenarios, not usually for beginner projects.
> 10. Fusion: Fusion combines compatible operations to reduce intermediate allocations and improve performance. It is an execution optimization rather than a completely separate hardware target.
> 11. Autodiff: Adds automatic differentiation.

The shape sizes are checked at runtime. The rank is part of the Rust type, so a rank-2 tensor cannot accidentally be passed where a rank-3 tensor is required.

For the taxi model, the training backend is:

```rust
use burn::backend::{Autodiff, NdArray};

type TrainBackend = Autodiff<NdArray>;
```

`Autodiff<NdArray>` wraps the CPU backend with the graph needed for `loss.backward()`.

## 2. Tensor initialization: the methods you will use most

All examples below assume:

```rust
use burn::{
    backend::NdArray,
    tensor::{Distribution, Int, Tensor, TensorData, TensorCreationOptions},
};

type B = NdArray;
let device = Default::default();
```

### 2.1 `from_data`: explicit values

Use `from_data` when the values are already in a Rust array, vector, or `TensorData`.

```rust
let matrix = Tensor::<B, 2>::from_data(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    &device,
);

let values = vec![1.0_f32, 2.0, 3.0, 4.0];
let vector = Tensor::<B, 2>::from_data(
    TensorData::new(values, [2, 2]),
    &device,
);
```

Use `TensorData::new` when data comes from a file or iterator and you know the shape separately.

### 2.2 `from_floats`: readable floating-point data

`from_floats` is convenient when the input is clearly floating point:

```rust
let vector = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device);
let matrix = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
```

For most ML features, labels, weights, and activations, floating-point tensors are the right starting point.

### 2.3 `from_ints`: integer tensors

Use integer tensors for token IDs, class IDs, indexes, and other discrete values:

```rust
let labels = Tensor::<B, 2, Int>::from_ints([[0, 1], [2, 1]], &device);
```

Integer tensors are not interchangeable with floating-point tensors. Convert deliberately when an operation requires another kind:

```rust
let as_float = labels.float();
```

### 2.4 `zeros` and `zeros_like`

```rust
let zeros = Tensor::<B, 2>::zeros([2, 3], &device);
let same_shape = matrix.zeros_like();
```

Use these for initialization, padding, masks, accumulators, and test fixtures.

### 2.5 `ones` and `ones_like`

```rust
let ones = Tensor::<B, 2>::ones([2, 3], &device);
let same_shape = matrix.ones_like();
```

The triangular example from `src/bin/burn.rs` is:

```rust
let ones = Tensor::<B, 2>::ones([3, 3], &device);
let lower_triangle = ones.tril(0);
```

### 2.6 `full` and `full_like`

Use `full` when every element should have a chosen value:

```rust
let fives = Tensor::<B, 2>::full([2, 3], 5.0, &device);
let same_shape = matrix.full_like(-1.0);
```

This is useful for sentinel values, constant targets, and test inputs.

### 2.7 `empty`

```rust
let uninitialized = Tensor::<B, 2>::empty([2, 3], &device);
```

`empty` allocates storage without filling it with a meaningful value. Prefer `zeros`, `ones`, or `full` unless you have a specific performance-sensitive reason to use it. Never read an empty tensor expecting initialized values.

### 2.8 `random` and `random_like`

```rust
let uniform = Tensor::<B, 2>::random([2, 3], Distribution::Uniform(0.0, 1.0), &device);
let normal = Tensor::<B, 2>::random([2, 3], Distribution::Normal(0.0, 1.0), &device);
let random_with_existing_shape = matrix.random_like(Distribution::Normal(0.0, 1.0));
```

Random initialization is common for experiments and model parameters. Reproducibility requires controlling the random seed or using deterministic initialization where appropriate. Burn's neural-network layers provide their own sensible initializers, so do not manually replace them unless you understand why.

### 2.9 `arange` and `arange_step`

`arange` creates integer sequences:

```rust
let positions = Tensor::<B, 1, Int>::arange(0..5, &device);
let every_second = Tensor::<B, 1, Int>::arange_step(0..10, 2, &device);
```

This is useful for positions, indexes, and batch construction. In the GPT example, positions are reshaped into `[1, time]` before being passed to an embedding.

### 2.10 `eye`

`eye` creates an identity matrix:

```rust
let identity = Tensor::<B, 2>::eye(3, &device);
```

Identity matrices are useful in linear algebra, residual connections, and tests.

### 2.11 Creation options

Most examples use `&device` because it is concise. Burn also accepts `TensorCreationOptions` when you need explicit device or dtype control:

```rust
let options = TensorCreationOptions::<B>::new(device.clone());
let tensor = Tensor::<B, 2>::ones([2, 2], options);
```

Use explicit options when a function accepts configuration from several places or when you need to document a dtype choice.

---

## 3. Inspecting tensors without noisy metadata

Burn's `Display` implementation intentionally prints values plus shape, device, backend, kind, and dtype:

```rust
println!("{tensor}");
```

To read only the values:

```rust
let values = tensor
    .into_data()
    .to_vec::<f32>()
    .expect("tensor data should be readable");
println!("{values:?}");
```

For a matrix, preserve rows yourself:

```rust
let values = matrix.into_data().to_vec::<f32>().unwrap();
for row in values.chunks(3) {
    println!("{row:?}");
}
```

Useful inspection methods:

```rust
println!("shape: {:?}", matrix.dims());
println!("rank: {}", matrix.dims().len());
println!("device: {:?}", matrix.device());
```

`into_data()` or `to_data()` may synchronize an accelerator with the CPU. Use them for logging, metrics, and final output, not on every operation in a hot training loop.

---

## 4. Shape and layout operations

Shape mistakes are among the most common ML bugs. Check shapes early:

```rust
let batch = Tensor::<B, 2>::ones([4, 3], &device);
assert_eq!(batch.dims(), [4, 3]);
```

### 4.1 `reshape`

`reshape` changes the view of the same number of elements:

```rust
let tensor = Tensor::<B, 2>::ones([2, 6], &device);
let reshaped: Tensor<B, 3> = tensor.reshape([2, 2, 3]);
let inferred: Tensor<B, 2> = reshaped.reshape([2, -1]);
```

The `-1` dimension is inferred. Burn will reject a shape whose element count does not match.

### 4.2 `flatten`

```rust
let images = Tensor::<B, 4>::ones([32, 1, 28, 28], &device);
let features: Tensor<B, 2> = images.flatten(1, 3);
assert_eq!(features.dims(), [32, 784]);
```

This is common before feeding images into a linear layer.

### 4.3 `unsqueeze` and `squeeze`

Add or remove dimensions of size one:

```rust
let vector = Tensor::<B, 1>::ones([3], &device);
let column: Tensor<B, 2> = vector.unsqueeze();
let restored: Tensor<B, 1> = column.squeeze();
```

Use `unsqueeze` to turn `[features]` into `[1, features]` for a one-item batch.

### 4.4 `transpose`, `t`, and `swap_dims`

```rust
let matrix = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
let transposed = matrix.clone().transpose(); // [3, 2]
let same = matrix.clone().t();                // alias for transpose
let swapped = matrix.swap_dims(0, 1);         // [3, 2]
```

For tensors with more than two dimensions, `transpose` swaps the last two dimensions. Use `swap_dims` when you want to name the dimensions explicitly.

### 4.5 `expand`

`expand` broadcasts a tensor to a larger compatible shape without writing a repeated copy in your source code:

```rust
let bias = Tensor::<B, 2>::ones([1, 3], &device);
let batch_bias: Tensor<B, 2> = bias.expand([4, 3]);
```

Use broadcasting for batch-wise constants, but verify that the expansion matches the intended semantics.

---

## 5. Element-wise arithmetic and mathematical operations

Burn overloads common operators:

```rust
let a = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
let b = Tensor::<B, 2>::ones([2, 2], &device);

let add = a.clone() + b.clone();
let subtract = a.clone() - b.clone();
let multiply = a.clone() * b.clone();
let divide = a / b;
```

Use scalar methods when one side is a Rust number:

```rust
let shifted = a.clone().add_scalar(1.0);
let scaled = a.clone().mul_scalar(2.0);
let divided = a.clone().div_scalar(2.0);
```

Common mathematical methods:

```rust
let absolute = a.clone().abs();
let bounded = a.clone().clamp(0.0, 1.0);
let lower_bounded = a.clone().clamp_min(0.0);
let upper_bounded = a.clone().clamp_max(1.0);
let squared = a.clone().powi_scalar(2);
let square_root = a.clone().sqrt();
let reciprocal = a.clone().recip();
let exponential = a.clone().exp();
let logarithm = a.log();
```

For neural networks, use Burn's activation functions:

```rust
use burn::tensor::activation;

let relu = activation::relu(a.clone());
let sigmoid = activation::sigmoid(a.clone());
let tanh = activation::tanh(a.clone());
let probabilities = activation::softmax(a, 1);
let log_probabilities = activation::log_softmax(probabilities, 1);
```

`relu`, `sigmoid`, `tanh`, `softmax`, and `log_softmax` are especially common in model `forward` methods.

### Matrix multiplication

```rust
let left = Tensor::<B, 2>::ones([4, 3], &device);
let right = Tensor::<B, 2>::ones([3, 2], &device);
let output = left.matmul(right);
assert_eq!(output.dims(), [4, 2]);
```

The inner dimensions must agree: `[batch, features] @ [features, outputs]`.

---

## 6. Reductions and statistics

Reductions turn many values into summaries:

```rust
let values = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

let total = values.clone().sum();
let average = values.clone().mean();
let minimum = values.clone().min();
let maximum = values.clone().max();
```

Reduce along a dimension when you need to preserve the rest of the shape:

```rust
let column_means = values.clone().mean_dim(0); // shape [1, 2]
let row_sums = values.clone().sum_dim(1);      // shape [2, 1]
let column_maximums = values.clone().max_dim(0);
let row_minimums = values.min_dim(1);
```

Other useful reductions include `argmax`, `min_abs`, `max_abs`, and variance/statistics helpers. Check the resulting shape after each reduction; whether a dimension is retained often matters to the next layer.

---

## 7. Indexing, slicing, concatenating, and masking

### 7.1 Slice ranges

```rust
let batch = Tensor::<B, 2>::from_floats(
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    &device,
);

let first_two_rows = batch.clone().slice([0..2, 0..2]);
let last_column = batch.clone().slice([0..3, 1..2]);
```

### 7.2 Select rows by integer indexes

```rust
let indexes = Tensor::<B, 1, Int>::from_ints([2, 0], &device);
let selected = batch.select(0, indexes);
```

### 7.3 Concatenate and stack

`cat` joins tensors along an existing dimension:

```rust
let first = Tensor::<B, 2>::ones([2, 3], &device);
let second = Tensor::<B, 2>::zeros([2, 3], &device);
let joined = Tensor::cat(vec![first, second], 0); // [4, 3]
```

`stack` creates a new dimension:

```rust
let a = Tensor::<B, 2>::ones([2, 3], &device);
let b = Tensor::<B, 2>::zeros([2, 3], &device);
let stacked: Tensor<B, 3> = Tensor::stack(vec![a, b], 0); // [2, 2, 3]
```

Use `chunk` to split a tensor:

```rust
let parts = joined.chunk(2, 0);
```

### 7.4 Triangular masks and `mask_fill`

```rust
use burn::tensor::Bool;

let mask = Tensor::<B, 2, Bool>::tril_mask([3, 3], 0, &device);
let scores = Tensor::<B, 2>::ones([3, 3], &device);
let masked = scores.mask_fill(mask, f32::NEG_INFINITY);
```

For causal attention, `tril_mask` keeps the current and previous positions visible. `triu_mask` creates the opposite triangular mask. `diag_mask` creates a diagonal mask. `tril` and `triu` directly replace values outside the selected triangle with zero.

---

## 8. Ownership: why `clone()` appears in Burn code

Tensor operations usually consume `self`:

```rust
let scaled = tensor.mul_scalar(2.0);
// tensor is moved and cannot be used here
```

When the same tensor is needed twice, clone it:

```rust
let left = tensor.clone() + other;
let right = tensor.clone().transpose();
```

This is a Rust ownership rule, not a Burn-specific requirement. In a training loop, avoid cloning more than necessary, but do not fight the type system. More importantly, avoid repeated `into_data()` calls: reading data back can be much more expensive than a device-side tensor operation.

---

## 9. Automatic differentiation

Use an autodiff backend when you need gradients:

```rust
use burn::{
    backend::{Autodiff, NdArray},
    tensor::Tensor,
};

type TrainBackend = Autodiff<NdArray>;
let device = Default::default();
let input = Tensor::<TrainBackend, 2>::from_floats([[2.0], [3.0]], &device);
let weight = Tensor::<TrainBackend, 2>::ones([1, 1], &device).require_grad();
let output = input.matmul(weight.clone());
let loss = output.clone().mean();
let gradients = loss.backward();
let weight_gradient = weight.grad(&gradients);
```

In practice, Burn modules and optimizers manage parameter gradients:

```rust
let prediction = model.forward(train_x.clone());
let loss = loss_fn.forward(prediction, train_y.clone(), Reduction::Mean);
let gradients = GradientsParams::from_grads(loss.backward(), &model);
model = optimizer.step(learning_rate, model, gradients);
```

That is the core of the `train` function in `taxis.rs`.

---

## 10. Building a Burn neural network

`TaxiModel` is intentionally small:

```text
[batch, 1] -> Linear(1, 8) -> ReLU -> Linear(8, 1) -> [batch, 1]
```

The Rust structure is:

```rust
#[derive(Module, Debug)]
struct TaxiModel<B: Backend> {
    hidden: Linear<B>,
    output: Linear<B>,
}

impl<B: Backend> TaxiModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            hidden: LinearConfig::new(1, 8).init(device),
            output: LinearConfig::new(8, 1).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.output
            .forward(activation::relu(self.hidden.forward(input)))
    }
}
```

The important design rules are:

1. `new` creates parameters on a device.
2. `forward` consumes an input tensor and returns the prediction tensor.
3. The input and output shapes are documented and tested.
4. The model is generic over `B`, so it is not permanently tied to one backend.

---

## 11. The taxi regression project

### Step 1: Load only the columns you need

Polars reads the Parquet file:

```rust
let frame = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?
    .select([col("trip_distance"), col("fare_amount")])
    .collect()?;
```

Do not load every column by default. Selecting the required columns reduces memory use and makes the feature contract clear.

### Step 2: Validate rows before creating tensors

The example keeps only finite, positive distances and fares:

```rust
if distance.is_finite()
    && fare.is_finite()
    && distance > 0.0
    && fare > 0.0
{
    distances.push(distance);
    fares.push(fare);
}
```

Invalid data should be handled at the boundary, not discovered as a mysterious training failure.

### Step 3: Split before fitting normalization

The example splits into training and validation data, then computes mean and standard deviation from training data only:

```text
normalized = (value - training_mean) / training_std
```

Using validation values to fit preprocessing is data leakage. The same training normalizers must be reused at validation and inference time.

### Step 4: Train and measure in understandable units

The model trains against normalized targets. The validation loss is converted back to dollar RMSE:

```rust
let validation_rmse = validation_loss.sqrt() * fare_normalizer.std;
```

Metrics should be understandable to the person using the model. `$7.11 RMSE` is more useful than `0.42 normalized MSE`.

### Step 5: Save and reload

```rust
let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
model.save_file("learning_burn/taxi_model", &recorder)?;

let loaded = TaxiModel::new(&device).load_file(
    "learning_burn/taxi_model",
    &recorder,
    &device,
)?;
```

Saving only weights is not enough for a production model. Save the feature names, normalization statistics, hyperparameters, Burn version, and model version alongside the weights.

---

## 12. Rust ML-engineering practices

- Keep data loading, validation, preprocessing, model code, training, and inference in separate functions.
- Return `Result` from reusable functions and add context to errors at the executable boundary.
- Use traits and generics to keep model code backend-independent.
- Test pure functions such as filtering, splitting, and normalization without loading the large dataset.
- Add shape tests for every model boundary.
- Keep training deterministic while learning; add seeded shuffling before comparing experiments.
- Do not read tensors back to CPU inside every training step.
- Run `cargo fmt`, `cargo check`, `cargo test`, and `cargo clippy`.
- Treat generated weights, logs, and plots as artifacts rather than source files.

---

## 13. Exercises

1. Replace the row-order split with a deterministic seeded shuffle.
2. Add a test that verifies the model output shape is `[batch, 1]`.
3. Add `passenger_count` as a second input feature and change `LinearConfig::new(1, 8)` to `LinearConfig::new(2, 8)`.
4. Add a held-out test split and report train, validation, and test RMSE.
5. Save and reload the normalizers together with the model.
6. Add mini-batches instead of using all training rows in one batch.
7. Add a command-line argument for the dataset path and number of epochs.
8. Compare a one-layer linear model with the two-layer model.
9. Try the same model with WGPU after the NdArray version is understood.
