// use burn::{backend::Wgpu, tensor::Tensor};
//
// type B = Wgpu<f32, i32>;
//
// fn main() {
//     let device = Default::default();
//     // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first
//     let tensor_1 = Tensor::<B, 2>::from_data([[2., 3.], [4., 5.]], &device);
//     let tensor_2 = Tensor::<B, 2>::ones_like(&tensor_1);
//
//     // Print the element-wise addition (done with the WGPU backend) of the two tensors.
//     println!("{}", tensor_1 + tensor_2);
// }

use burn::{backend::Wgpu, tensor::Tensor};

type B = Wgpu<f32, i32>;

fn main() {
    let device = Default::default();
    let ones = Tensor::<B, 2>::ones([3, 3], &device);
    let lower_triangle = ones.tril(0);

    println!("{}", lower_triangle);
}
