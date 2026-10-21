// use burn::{Tensor, backend::NdArray};
//
// type B = NdArray;
//
// fn main() {
//     let device = Default::default();
//
//     let miles = Tensor::<B, 2>::from_data([
//         [2.50, 9.37, 11.21, 2.48, 3.36, 7.14, 1.38, 0.88, 1.71, 1.00, 1.47, 1.26],
//         [0.88, 1.59, 11.40, 1.20, 1.60, 3.44, 1.85, 2.02, 0.60, 1.00, 0.63, 1.33],
//         [6.74, 2.88, 0.70, 6.60, 3.74, 2.19, 1.89, 1.95, 0.59, 1.19, 2.13, 4.21],
//     ], &device);
//
//     let fares = Tensor::<B, 2>::from_data([
//         [14.2, 38.0, 43.6, 15.6, 19.8, 45.7, 11.4, 7.9, 10.7, 8.6, 10.7, 9.3],
//         [5.8, 9.3, 44.3, 8.6, 12.1, 20.5, 13.5, 21.9, 10.0, 8.6, 6.5, 10.7],
//         [29.6, 18.4, 6.5, 33.8, 20.5, 11.4, 11.4, 12.8, 5.8, 9.3, 14.2, 21.2],
//     ], &device);
//
//     // let weights = Tensor::<B, 1>::from_data([1.00], &device);
//     // let bias = Tensor::<B,1>::from_data([0.00], &device);
//     let weights = 1.00;
//     let bias = 0.00;
//
//     let learning_rate = 0.0001;
//
//     saada_implementation(miles, fares, weights, bias, learning_rate)
// }
//
// fn saada_implementation(m: Tensor<B,2>, f: Tensor<B,2>, mut w: f32, mut b: f32, lr: f32) {
//     for i in 0..10000 {
//         let pred_fare = m.clone() * w + b;
//         let err = pred_fare - f.clone();
//         let loss = (err.clone()).powf_scalar(2.0).mean().into_scalar();
//         let dw = (err.clone() * 2.0 * m.clone()).mean().into_scalar();
//         let db = (err.clone() * 2.0).mean().into_scalar();
//
//         w = w - (dw * lr);
//         b = b - (db * lr);
//
//         println!(
//             " -[{i}]- Weight:{w:.2}, Bias:{b:.2}, Loss:{loss:.2}",
//         );
//     }
// }

use ndarray::{array, Array2};

fn main() {
    println!("Hello, World");
    let miles = array![
        [2.50, 9.37, 11.21, 2.48, 3.36, 7.14, 1.38, 0.88, 1.71, 1.00, 1.47, 1.26],
        [0.88, 1.59, 11.40, 1.20, 1.60, 3.44, 1.85, 2.02, 0.60, 1.00, 0.63, 1.33],
        [6.74, 2.88, 0.70, 6.60, 3.74, 2.19, 1.89, 1.95, 0.59, 1.19, 2.13, 4.21],
    ];

    let fares = array![
        [14.2, 38.0, 43.6, 15.6, 19.8, 45.7, 11.4, 7.9, 10.7, 8.6, 10.7, 9.3],
        [5.8, 9.3, 44.3, 8.6, 12.1, 20.5, 13.5, 21.9, 10.0, 8.6, 6.5, 10.7],
        [29.6, 18.4, 6.5, 33.8, 20.5, 11.4, 11.4, 12.8, 5.8, 9.3, 14.2, 21.2],
    ];

    let weights = 1.00;
    let bias = 0.00;
    let learning_rate = 0.0001;

    saada_implementation(miles, fares, weights, bias, learning_rate)
}

fn saada_implementation(m: Array2<f64>, f: Array2<f64>, mut w: f64, mut b: f64, lr: f64) {
    for i in 0..=10000 {
        let pred_fare = m.mapv(|value| value * w + b);
        let err = &pred_fare - &f;
        let sample_count = err.len() as f64;
        let loss = err.mapv(|value| value.powi(2)).sum() / sample_count;
        let dw = (&err * &m).sum() * 2.0 / sample_count;
        let db = err.sum() * 2.0 / sample_count;

        w = w - (dw * lr);
        b = b - (db * lr);

        println!("-[{i}]- Weight:{w:.2}, Bias:{b:.2}, Loss:{loss:.2}");
    }
}
