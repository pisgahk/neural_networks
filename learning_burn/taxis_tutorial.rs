use std::{
    error::Error,
    path::{Path, PathBuf},
};

use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    nn::{
        Linear, LinearConfig,
        loss::{MseLoss, Reduction},
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Tensor, TensorData, activation, backend::Backend},
};
use polars::prelude::*;

pub type TrainBackend = Autodiff<NdArray>;

const INPUT_FEATURES: usize = 1;
const HIDDEN_FEATURES: usize = 8;
const EPOCHS: usize = 80;
const LEARNING_RATE: f64 = 1e-2;
const DEMO_ROWS: usize = 20_000;

#[derive(Clone, Debug, PartialEq)]
pub struct Normalizer {
    pub mean: f32,
    pub std: f32,
}

impl Normalizer {
    pub fn fit(values: &[f32]) -> Result<Self, Box<dyn Error>> {
        if values.is_empty() {
            return Err("cannot normalize an empty slice".into());
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / values.len() as f32;
        let std = variance.sqrt();

        if !std.is_finite() || std <= f32::EPSILON {
            return Err("normalization requires non-constant finite values".into());
        }

        Ok(Self { mean, std })
    }

    pub fn transform(&self, values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .map(|value| (value - self.mean) / self.std)
            .collect()
    }

    pub fn inverse(&self, values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .map(|value| value * self.std + self.mean)
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TaxiDataset {
    pub distances: Vec<f32>,
    pub fares: Vec<f32>,
}

impl TaxiDataset {
    pub fn from_pairs(pairs: impl IntoIterator<Item = (f32, f32)>) -> Result<Self, Box<dyn Error>> {
        let mut distances = Vec::new();
        let mut fares = Vec::new();

        for (distance, fare) in pairs {
            if distance.is_finite() && fare.is_finite() && distance > 0.0 && fare > 0.0 {
                distances.push(distance);
                fares.push(fare);
            }
        }

        if distances.len() < 2 {
            return Err("dataset must contain at least two valid rows".into());
        }

        Ok(Self { distances, fares })
    }

    pub fn split(&self, train_fraction: f32) -> Result<(Self, Self), Box<dyn Error>> {
        if !(0.0..1.0).contains(&train_fraction) {
            return Err("train fraction must be between 0 and 1".into());
        }

        let split = ((self.distances.len() as f32) * train_fraction).round() as usize;
        if split == 0 || split == self.distances.len() {
            return Err("train split must leave rows for both train and validation".into());
        }

        let train = Self::from_pairs(
            self.distances[..split]
                .iter()
                .copied()
                .zip(self.fares[..split].iter().copied()),
        )?;
        let validation = Self::from_pairs(
            self.distances[split..]
                .iter()
                .copied()
                .zip(self.fares[split..].iter().copied()),
        )?;
        Ok((train, validation))
    }
}

pub fn load_parquet(path: impl AsRef<Path>) -> Result<TaxiDataset, Box<dyn Error>> {
    let path_string = path.as_ref().to_string_lossy().into_owned();
    let frame = LazyFrame::scan_parquet(path_string.as_str().into(), ScanArgsParquet::default())?
        .select([col("trip_distance"), col("fare_amount")])
        .collect()?;
    dataset_from_frame(&frame)
}

pub fn load_csv(path: impl AsRef<Path>) -> Result<TaxiDataset, Box<dyn Error>> {
    let frame = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
        .finish()?;
    dataset_from_frame(&frame)
}

fn dataset_from_frame(frame: &DataFrame) -> Result<TaxiDataset, Box<dyn Error>> {
    let distances = frame.column("trip_distance")?.f64()?;
    let fares = frame.column("fare_amount")?.f64()?;
    Ok(TaxiDataset::from_pairs(
        distances
            .into_iter()
            .zip(fares)
            .filter_map(|(distance, fare)| Some((distance? as f32, fare? as f32))),
    )?)
}

#[derive(Module, Debug)]
pub struct TaxiModel<B: Backend> {
    hidden: Linear<B>,
    output: Linear<B>,
}

impl<B: Backend> TaxiModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            hidden: LinearConfig::new(INPUT_FEATURES, HIDDEN_FEATURES).init(device),
            output: LinearConfig::new(HIDDEN_FEATURES, 1).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.output
            .forward(activation::relu(self.hidden.forward(input)))
    }
}

pub fn tensor_from_values<B: Backend>(values: &[f32], device: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(values.to_vec(), [values.len(), 1]), device)
}

pub fn train(
    train: &TaxiDataset,
    validation: &TaxiDataset,
    epochs: usize,
    learning_rate: f64,
    device: &<TrainBackend as Backend>::Device,
) -> Result<(TaxiModel<TrainBackend>, f32), Box<dyn Error>> {
    let distance_normalizer = Normalizer::fit(&train.distances)?;
    let fare_normalizer = Normalizer::fit(&train.fares)?;
    let train_x = tensor_from_values::<TrainBackend>(
        &distance_normalizer.transform(&train.distances),
        device,
    );
    let train_y =
        tensor_from_values::<TrainBackend>(&fare_normalizer.transform(&train.fares), device);
    let validation_x = tensor_from_values::<TrainBackend>(
        &distance_normalizer.transform(&validation.distances),
        device,
    );
    let validation_y =
        tensor_from_values::<TrainBackend>(&fare_normalizer.transform(&validation.fares), device);

    let mut model = TaxiModel::new(device);
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = MseLoss::new();

    for _ in 0..epochs {
        let prediction = model.forward(train_x.clone());
        let loss = loss_fn.forward(prediction, train_y.clone(), Reduction::Mean);
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(learning_rate, model, gradients);
    }

    let validation_prediction = model.forward(validation_x);
    let validation_loss = loss_fn
        .forward(validation_prediction, validation_y, Reduction::Mean)
        .into_data()
        .to_vec::<f32>()?[0];
    let validation_rmse = validation_loss.sqrt() * fare_normalizer.std;
    Ok((model, validation_rmse))
}

pub fn predict(
    model: &TaxiModel<TrainBackend>,
    distances: &[f32],
    distance_normalizer: &Normalizer,
    fare_normalizer: &Normalizer,
    device: &<TrainBackend as Backend>::Device,
) -> Vec<f32> {
    let input =
        tensor_from_values::<TrainBackend>(&distance_normalizer.transform(distances), device);
    let normalized = model
        .forward(input)
        .into_data()
        .to_vec::<f32>()
        .expect("prediction tensor should be readable");
    fare_normalizer.inverse(&normalized)
}

pub fn save_model(
    model: TaxiModel<TrainBackend>,
    path: impl Into<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    model.save_file(path, &recorder)?;
    Ok(())
}

pub fn load_model(
    path: impl Into<PathBuf>,
    device: &<TrainBackend as Backend>::Device,
) -> Result<TaxiModel<TrainBackend>, Box<dyn Error>> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    Ok(TaxiModel::new(device).load_file(path, &recorder, device)?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let dataset_path = Path::new("learning_burn/yellow_tripdata_2024-01.parquet");
    let full_dataset = load_parquet(dataset_path)?;
    let dataset = TaxiDataset::from_pairs(
        full_dataset
            .distances
            .iter()
            .copied()
            .zip(full_dataset.fares.iter().copied())
            .take(DEMO_ROWS),
    )?;
    let (train_data, validation_data) = dataset.split(0.8)?;
    let distance_normalizer = Normalizer::fit(&train_data.distances)?;
    let fare_normalizer = Normalizer::fit(&train_data.fares)?;
    let device = Default::default();

    println!(
        "loaded {} valid rows ({} train, {} validation)",
        dataset.distances.len(),
        train_data.distances.len(),
        validation_data.distances.len()
    );

    let (model, validation_rmse) = train(
        &train_data,
        &validation_data,
        EPOCHS,
        LEARNING_RATE,
        &device,
    )?;
    let model_path = Path::new("learning_burn/taxi_model");
    save_model(model, model_path)?;
    let saved_model = load_model(model_path, &device)?;

    let examples = [1.0, 3.0, 10.0];
    let predictions = predict(
        &saved_model,
        &examples,
        &distance_normalizer,
        &fare_normalizer,
        &device,
    );
    println!("validation RMSE: ${validation_rmse:.2}");
    for (distance, fare) in examples.into_iter().zip(predictions) {
        println!("{distance:.1} miles -> predicted fare ${fare:.2}");
    }
    println!("saved model to learning_burn/taxi_model.bin");
    Ok(())
}
