#[path = "../learning_burn/taxis.rs"]
mod taxis;

use burn::backend::NdArray;
use taxis::{
    Normalizer, TaxiDataset, load_csv, load_model, predict, save_model, tensor_from_values, train,
};

#[test]
fn filters_invalid_taxi_rows() {
    let dataset = TaxiDataset::from_pairs([
        (1.0, 5.0),
        (0.0, 4.0),
        (2.0, -1.0),
        (f32::NAN, 3.0),
        (3.0, 9.0),
    ])
    .expect("valid rows should remain");

    assert_eq!(dataset.distances, vec![1.0, 3.0]);
    assert_eq!(dataset.fares, vec![5.0, 9.0]);
}

#[test]
fn normalization_round_trips_values() {
    let values = [10.0, 20.0, 30.0];
    let normalizer = Normalizer::fit(&values).expect("values are not constant");
    let normalized = normalizer.transform(&values);
    let restored = normalizer.inverse(&normalized);

    for (actual, expected) in restored.iter().zip(values) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[test]
fn fixture_can_be_loaded() {
    let dataset = load_csv("learning_burn/fixtures/taxi_small.csv").expect("fixture should load");
    assert_eq!(dataset.distances.len(), 8);
    assert_eq!(dataset.fares[0], 5.5);
}

#[test]
fn tensors_have_one_column_per_feature() {
    let device = <NdArray as burn::tensor::backend::Backend>::Device::default();
    let tensor = tensor_from_values::<NdArray>(&[1.0, 2.0, 3.0], &device);

    assert_eq!(tensor.dims(), [3, 1]);
}

#[test]
fn model_can_be_saved_and_loaded() {
    let dataset = load_csv("learning_burn/fixtures/taxi_small.csv").expect("fixture should load");
    let (train_data, validation_data) = dataset.split(0.75).expect("split should succeed");
    let distance_normalizer = Normalizer::fit(&train_data.distances).expect("distances vary");
    let fare_normalizer = Normalizer::fit(&train_data.fares).expect("fares vary");
    let device = <taxis::TrainBackend as burn::tensor::backend::Backend>::Device::default();
    let (model, _) = train(&train_data, &validation_data, 2, 1e-2, &device).expect("training");
    let path = "target/taxi_test_model";

    save_model(model, path).expect("model should save");
    let loaded = load_model(path, &device).expect("model should load");
    let predictions = predict(
        &loaded,
        &[1.0, 2.0],
        &distance_normalizer,
        &fare_normalizer,
        &device,
    );

    assert_eq!(predictions.len(), 2);
    let _ = std::fs::remove_file("target/taxi_test_model.bin");
}
