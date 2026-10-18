use ndarray::prelude::*;
use ndarray_rand::{
    RandomExt,
    rand::{SeedableRng, rngs::StdRng, seq::SliceRandom},
    rand_distr::Uniform,
};

fn main() {
    // let a = array!["A", "B", "C"];
    let a = array![
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.],
        [13., 14., 15., 16.]
    ];
    // assert_eq!(a.ndim(), 2);         // get the number of dimensions of array a
    // assert_eq!(a.len(), 6);          // get the number of elements in array a
    // assert_eq!(a.shape(), [2, 3]);   // get the shape of array a
    // assert!(a.is_empty(), {false); // check if the array has zero elements
    // let mut a = array![1,2,3,4]; a *= 2;

    let name = &a.slice(s![0..=1, 1..=2]); // Basic slicing
    let name2 = &a.slice(s![0..2 , 2..=3 ; 1]); //Slicing with steps

    // Arithmatic.
    // let line_one = array![1.,2.,3.];
    let line_one = a.slice(s![0, ..3]);
    let line_first = &line_one + 1.0;
    let line_two = &line_one - 3.0;
    let line_three = &line_one * 3.0;
    let line_four = &line_one / 4.0;

    println!("a is: {:?}", a);
    println!("{}", a.ndim());
    println!("name is: {}", name);
    println!("name2 is: {}", name2);

    println!("Scalar addition: {}", line_first);

    println!("Scalar subtraction: {}", line_two);
    println!("Scalar multiplication: {}", line_three);
    println!("Scalar division: {}", line_four);

    // ---[Comparison Arithmatic/Filtering]----------------------------------

    let arr1 = array![1., 2., 3.];
    let arr2 = array![4., 5., 6.];
    let arr3 = array![[78, 12, 55, 43, 99, 84, 86], [23, 33, 62, 3, 2, 23, 18]];

    let total = &arr1 / &arr2;
    let comparison_arithmetic = &arr3.mapv(|s| if s <= 44 { -1 } else { s }); // Comparison Arithmetic. The output returns a
    // boolean array.
    let under_sixty_bool = &arr3 // Comparison Arithmatic: Boolean Array
        .mapv(|s| s <= 60);

    let under_sixty_values = &arr3 // Comparison Arithmatic: Value Array
        .iter()
        .filter(|&&s| s < 60)
        .copied()
        .collect::<Array1<i32>>();

    let under_sixtyfive_greaterthan_eighteen = &arr3
        .iter()
        .filter(|&&s| s > 18 && s < 65)
        .copied()
        .collect::<Array1<i32>>();

    let evens = &arr3
        .iter()
        .filter(|&&s| s % 2 == 0)
        .copied()
        .collect::<Array1<i32>>();
    let odds = &arr3
        .iter()
        .filter(|&&s| s % 2 != 0)
        .copied()
        .collect::<Array1<i32>>();

    let dot_pdt = arr1.dot(&arr2);

    println!("Element-wise addition: {}", total);
    println!("Comparison Arithmetic: {:?}", comparison_arithmetic);
    println!("Dot Product: {}", dot_pdt);
    println!("Under 60(boolean): {}", under_sixty_bool);
    println!("Under 60(values): {}", under_sixty_values);
    println!(
        "under_sixtyfive_greaterthan_eighteen: {}",
        under_sixtyfive_greaterthan_eighteen
    );
    println!("Even Ages: {}", evens);
    println!("Odd Ages: {:?}", odds);

    //     Here's the condition, in short: shape-preserving and filtering are mutually exclusive operations, and the tool you reach for depends on which one you actually want.
    //
    // If you want the same shape as the input, with non-matching spots replaced by something (zero, `None`, or the original value), use `mapv` with an `if`/`else` (or `Option`). This is the direct equivalent of NumPy's three-argument `np.where(condition, x, y)`.
    //
    // If you want just the matching values, with no concern for original position or shape, use `.iter().filter(...).copied().collect::<Array1<_>>()`. This collapses to 1D, and it's actually what NumPy's `arr[mask]` is doing too, even though the syntax disguises it.
    //
    // If you want to know which positions matched without discarding anything, use `mapv` returning `bool`, producing a same-shaped boolean mask you can pair with the original array.
    //
    // There's no single idiomatic call that filters and preserves shape simultaneously, in `ndarray` or in NumPy — every apparent exception is really one of the two paths above wearing different clothes.

    //     | Goal | Method | Shape |
    //     |---|---|---|
    //     | Replace non-matches, keep shape | `mapv(\|s\| if cond {s} else {0})` | Same as input |
    //     | Just the matching values | `iter().filter().collect::<Array1<_>>()` | Flattened, shorter |
    //     | Know which positions match | `mapv(\|s\| cond)` → bool mask | Same as input |
    //
    // No single call does both filter and shape-preserve — NumPy's `arr[mask]` flattens too, just hides it.

    // ---[Broadcasting in ndarray]------------------------------------------------------------------

    // Rules for braodcasting:
    // - They are identical in size.
    // - One of them is exaclty 1.

    let matrix = array![[1., 2., 3.], [4., 5., 6.]];
    let row_vector = array![[10.], [20.]];

    println!("matrix shape is: {:?}", matrix.shape());
    println!("row_vector shape is: {:?}", row_vector.shape());

    let broadcasted_row = &row_vector.broadcast((2, 3)).unwrap();

    let result = &matrix + broadcasted_row;
    println!("Broadcasting: {}", result);

    // Creating math table

    let side_one = array![[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]];
    let side_two = array![[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.], [10.]];
    println!("{:?}", side_one.shape());
    println!("{:?}", side_two.shape());

    let mul_table = &side_one * &side_two;

    println!("{mul_table}");

    // ---[Aggregate functions]-------------------------------

    let array_one = array![[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]];

    println!("Sum_Axis is: {}", array_one.sum_axis(Axis(1))); // sum, product, mean,
    // sum_axis(Axis(0) for all columns.
    println!("Variance is: {}", array_one.var(0.)); // Use 0 for simple var and 1 for sample variance. 
    println!("Standard Deviation is: {}", array_one.std(0.)); // Use 0 for simple stdev and 1 for sample stdev.

    // ---[Random Number Generation]----------------------------------------------

    let distribution = Uniform::new(1., 10.).expect("Check distribution ranges");
    // You can also:
    // Uniform::new()
    // Exp::new()
    // LogNormal::new()
    // Binomial::new()
    // Bernoulli::new()
    // Poisson::new()
    //
    // Checkout ndarray-rand::rand_distr::<Structs>

    let arr = Array2::<f32>::random((3, 4), &distribution); // Unseeded random
    println!("Random array: {}", arr);

    let mut rng = StdRng::seed_from_u64(43);
    let arr_seeds = Array2::<f32>::random_using((4, 5), &distribution, &mut rng); // Seeded random
    println!("Seeded Random numbers: {}", arr_seeds);

    // let my_rng = rng();
    let mut my_rng = StdRng::try_from_os_rng().unwrap();

    // Shuffling arrays. Read more about this.
    let mut shuffle_array = Array2::<f32>::random((4, 7), &distribution);
    shuffle_array.as_slice_mut().unwrap().shuffle(&mut my_rng);
    println!("Shuffle Array: {:#?}", shuffle_array);
}
