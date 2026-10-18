use ndarray::prelude::*;
use ndarray::{Array, Array2, Array3, ShapeBuilder};
use ndarray_rand::{
    RandomExt,
    rand::{SeedableRng, rngs::StdRng, seq::SliceRandom},
    rand_distr::Uniform,
};

fn main() {
    // ---[Array Basics & Inspection]------------------------------------------
    println!("\n---[Array Basics & Inspection]------------------------------------------");

    let a = array![
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.],
        [13., 14., 15., 16.]
    ];

    println!("a is: \n {:?}", a);
    println!("ndim: {}", a.ndim()); // number of dimensions
    println!("len: {}", a.len()); // total number of elements
    println!("shape: {:?}", a.shape());
    println!("is_empty: {}", a.is_empty());

    // ---[Array Creation]------------------------------------------------------
    println!("\n---[Array Creation]------------------------------------------------------");
    // Multiple ways to construct arrays beyond the array![] macro.

    // From a flat Vec, specifying shape manually
    let from_vec = Array::from_vec(vec![1., 2., 3., 4.]);
    println!("from_vec: {}", from_vec);

    let from_shape_vec = Array::from_shape_vec((2, 2), vec![1., 2., 3., 4.]).unwrap();
    println!("from_shape_vec (2x2): {:?}", from_shape_vec);

    // Typed aliases: Array2, Array3 etc. are just Array<T, Ix2>, Array<T, Ix3>
    let array2 = Array2::from_shape_vec((2, 2), vec![1., 2., 3., 4.]).unwrap();
    let array3 = Array3::from_shape_vec((2, 2, 1), vec![1., 2., 3., 4.]).unwrap();
    println!("Array2: {}", array2);
    println!("Array3: {}", array3);

    // ---[Initial Placeholders]------------------------------------------------
    println!("\n---[Initial Placeholders]------------------------------------------------");
    // Equivalent of numpy's np.zeros, np.ones, np.arange, np.linspace, np.eye
    // .f() on a shape converts it to column-major (Fortran) order.
    // Without .f(), the default is row-major (C) order.

    let zeros = Array::<f64, _>::zeros((2, 4).f());
    println!("zeros with .f(): {:?}", zeros);

    let zeros_mod = Array::<f64, _>::zeros((2, 4));
    println!("zeros without .f(): {:?}", zeros_mod);

    let ones = Array::<f64, _>::ones((2, 4));
    println!("ones: {:?}", ones);

    // range(start, end_exclusive, step) -- equivalent of np.arange
    let range = Array::<f64, _>::range(0., 5., 1.);
    println!("range: {:?}", range);

    // linspace(start, end_inclusive, n) -- n evenly spaced values
    let linspace = Array::<f64, _>::linspace(0., 5., 5);
    println!("linspace: {:?}", linspace);

    // fill() mutates in-place, replacing every element with a fixed value
    let mut filled = Array::<f64, _>::ones((2, 4));
    filled.fill(7.);
    println!("filled: {:?}", filled);

    // eye(n) -- identity matrix: 1s on diagonal, 0s elsewhere
    let eye = Array::<f64, _>::eye(4);
    println!("eye: {:?}", eye);

    // ---[Slicing]------------------------------------------------------------
    println!("\n---[Slicing]------------------------------------------------------------");

    let basic_slice = a.slice(s![0..=1, 1..=2]); // rows 0-1, cols 1-2
    let stepped_slice = a.slice(s![0..2, 2..=3; 1]); // slicing with a step

    println!("basic_slice: {}", basic_slice);
    println!("stepped_slice: {}", stepped_slice);

    // ---[Reshaping]----------------------------------------------------------
    println!("\n---[Reshaping]----------------------------------------------------------");
    // reshape() is only available on ArcArray (rcarr1/rcarr2), not plain Array.
    // It changes shape without changing the data or total element count.
    // into_shape() is the equivalent for owned Arrays -- it also requires the
    // total element count to stay the same.

    // let arc_arr = rcarr1(&[1., 2., 3., 4.]);  // creates an ArcArray1
    // let reshaped = arc_arr.reshape((2, 2)); // Deprecated method reshape()
    // println!("reshaped (2x2): {:?}", reshaped);

    // For owned Array, use into_shape_with_order:
    let owned = array![1., 2., 3., 4., 5., 6.];
    let reshaped_owned = owned.into_shape_with_order((2, 3)).unwrap();
    println!("into_shape_with_order (2x3): {:?}", reshaped_owned);

    // Flatten: collapse any shape into a 1D array (always allocates a new copy)
    // Requires ndarray-linalg: use ndarray_linalg::convert::flatten;
    // let flat = flatten(some_2d_array);

    // ---[Transposing]--------------------------------------------------------
    println!("\n---[Transposing]--------------------------------------------------------");
    // .t() returns a VIEW of the array with axes reversed -- no data is copied.
    // The result has strides in the opposite order (Fortran layout vs C layout).

    let mat = Array::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap();
    println!("original: {:?}", mat);
    println!("transposed: {:?}", mat.t());

    // ---[Swapping Axes]------------------------------------------------------
    println!("\n---[Swapping Axes]------------------------------------------------------");
    // swap_axes(a, b) swaps two specific axes in-place (mutates the array).
    // Unlike .t() which reverses ALL axes, swap_axes lets you target a pair.
    // Useful for 3D+ arrays where you want to permute one pair of dimensions.

    let mut swappable =
        Array::from_shape_vec((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap();
    swappable.swap_axes(0, 1); // equivalent to .t() for 2D
    println!("swap_axes(0,1): {:?}", swappable);

    // ---[Scalar Arithmetic]--------------------------------------------------
    println!("\n---[Scalar Arithmetic]--------------------------------------------------");

    let line_one = a.slice(s![0, ..3]);

    println!("Scalar addition: {}", &line_one + 1.0);
    println!("Scalar subtraction: {}", &line_one - 3.0);
    println!("Scalar multiplication: {}", &line_one * 3.0);
    println!("Scalar division: {}", &line_one / 4.0);

    // ---[Comparison Arithmetic / Filtering]---------------------------------
    println!("\n---[Comparison Arithmetic / Filtering]---------------------------------");
    // Shape-preserving vs. filtering are mutually exclusive operations:
    //   - mapv(|s| if cond {s} else {sentinel}) -> same shape as input
    //     (this is the equivalent of NumPy's np.where(cond, x, y))
    //   - iter().filter(...).collect::<Array1<_>>() -> flattened, shorter
    //     (this is what NumPy's arr[mask] is *actually* doing too)
    //   - mapv(|s| cond) -> bool mask, same shape as input
    // There is no single call that filters AND preserves shape at once.

    let arr1 = array![1., 2., 3.];
    let arr2 = array![4., 5., 6.];
    let arr3 = array![[78, 12, 55, 43, 99, 84, 86], [23, 33, 62, 3, 2, 23, 18]];

    let total = &arr1 / &arr2;
    let dot_pdt = arr1.dot(&arr2);

    // Same-shape replace (np.where(cond, x, y) equivalent)
    let comparison_arithmetic = arr3.mapv(|s| if s <= 44 { -1 } else { s });

    // Same-shape boolean mask
    let under_sixty_bool = arr3.mapv(|s| s <= 60);

    // Flattened: matching values only
    let under_sixty_values = arr3
        .iter()
        .filter(|&&s| s < 60)
        .copied()
        .collect::<Array1<i32>>();
    let between_18_and_65 = arr3
        .iter()
        .filter(|&&s| s > 18 && s < 65)
        .copied()
        .collect::<Array1<i32>>();
    let evens = arr3
        .iter()
        .filter(|&&s| s % 2 == 0)
        .copied()
        .collect::<Array1<i32>>();
    let odds = arr3
        .iter()
        .filter(|&&s| s % 2 != 0)
        .copied()
        .collect::<Array1<i32>>();

    println!("Element-wise division: {}", total);
    println!("Dot product: {}", dot_pdt);
    println!(
        "Comparison arithmetic (same shape, sentinel -1): {:?}",
        comparison_arithmetic
    );
    println!("Under 60 (bool mask, same shape): {}", under_sixty_bool);
    println!("Under 60 (values, flattened): {}", under_sixty_values);
    println!("Between 18 and 65 (flattened): {}", between_18_and_65);
    println!("Evens (flattened): {}", evens);
    println!("Odds (flattened): {:?}", odds);

    // ---[Broadcasting]------------------------------------------------------
    println!("\n---[Broadcasting]------------------------------------------------------");
    // Rules for broadcasting (same as NumPy):
    //   - Dimensions are identical in size, OR
    //   - One of the dimensions is exactly 1.

    let matrix = array![[1., 2., 3.], [4., 5., 6.]];
    let row_vector = array![[10.], [20.]];

    println!("matrix shape: {:?}", matrix.shape());
    println!("row_vector shape: {:?}", row_vector.shape());

    let broadcasted_row = row_vector.broadcast((2, 3)).unwrap();
    println!("Broadcasted row: {}", &broadcasted_row);
    println!("Broadcasted Addition: {}", &matrix + &broadcasted_row);

    // Example: building a multiplication table via broadcasting
    let side_one = array![[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]];
    let side_two = array![[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.], [10.]];

    println!("side_one shape: {:?}", side_one.shape());
    println!("side_two shape: {:?}", side_two.shape());
    println!("Multiplication table:\n{}", &side_one * &side_two);

    // ---[Aggregate Functions]-----------------------------------------------
    println!("\n---[Aggregate Functions]-----------------------------------------------");

    let array_one = array![[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]];

    // Axis(0) aggregates down columns, Axis(1) aggregates across rows.
    // sum, product, mean all follow the same axis convention.
    println!("Sum along Axis(1): {}", array_one.sum_axis(Axis(1)));
    println!("Mean: {:?}", array_one.mean().unwrap());

    // ddof: 0 for population variance/stdev, 1 for sample variance/stdev
    println!("Variance (population): {}", array_one.var(0.));
    println!("Std dev (population): {}", array_one.std(0.));

    // ---[Random Number Generation]------------------------------------------
    println!("\n---[Random Number Generation]------------------------------------------");
    // ndarray itself has no RNG -- randomness comes from `ndarray-rand`,
    // which bridges `ndarray` with the `rand` / `rand_distr` crates.
    //
    // Other distributions available via ndarray_rand::rand_distr:
    //   Uniform::new(), Normal::new(), Exp::new(), LogNormal::new(),
    //   Binomial::new(), Bernoulli::new(), Poisson::new()
    // As of newer rand_distr versions, these constructors return
    // Result<Dist, Error> rather than Dist directly, so they need
    // .unwrap() or .expect() at the call site.

    let distribution = Uniform::new(1., 10.).expect("invalid range for Uniform distribution");

    // Unseeded: different output every run (seeded from OS entropy)
    let arr = Array2::<f32>::random((3, 4), &distribution);
    println!("Random array (unseeded): {}", arr);

    // Seeded: identical output every run for the same seed -> reproducible
    let mut rng = StdRng::seed_from_u64(43);
    let arr_seeded = Array2::<f32>::random_using((4, 5), &distribution, &mut rng);
    println!("Random array (seeded, reproducible): {}", arr_seeded);

    // ---[Shuffling]---------------------------------------------------------
    println!("\n---[Shuffling]---------------------------------------------------------");
    // ndarray has no shuffle of its own -- drop down to a mutable slice via
    // as_slice_mut() and use rand::seq::SliceRandom::shuffle().
    // as_slice_mut() returns None for non-contiguous arrays, hence unwrap().

    let mut shuffle_rng = StdRng::try_from_os_rng().unwrap();
    let mut shuffle_array = Array2::<f32>::random((4, 7), &distribution);
    shuffle_array
        .as_slice_mut()
        .unwrap()
        .shuffle(&mut shuffle_rng);
    println!("Shuffled array: {:#?}", shuffle_array);
}

// ---[Linear Algebra (ndarray-linalg)]--------------------------------------
// These operations require the `ndarray-linalg` crate plus a BLAS backend
// (e.g. `blas-src` with openblas feature). Add to Cargo.toml:
//
//   ndarray-linalg = { version = "0.x", features = ["openblas-static"] }
//   extern crate blas_src; // at top of main.rs
//
// Imports needed:
//   use ndarray_linalg::{
//       solve::Inverse,       // .inv()
//       solve::Determinant,   // .det()
//       trace::Trace,         // .trace()
//       Eig,                  // .eig()  eigenvalues + eigenvectors
//       SVD,                  // .svd()  singular value decomposition
//       Solve,                // .solve_into() for ax=b systems
//   };
//
// ---[Matrix Inversion]---
// let a: Array2<f64> = array![[1., 2.], [2., 1.]];
// match a.inv() {
//     Ok(inv) => println!("Inverse: {}", inv),
//     Err(e)  => println!("Error: {e}"),
// }
//
// ---[Determinant]---
// match a.det() {
//     Ok(d)  => println!("Det: {d}"),
//     Err(e) => println!("Error: {e}"),
// }
//
// ---[Trace (sum of diagonal)]---
// match a.trace() {
//     Ok(t)  => println!("Trace: {t}"),
//     Err(e) => println!("Error: {e}"),
// }
//
// ---[Eigenvalues & Eigenvectors]---
// match a.eig() {
//     Ok((eigenvalues, eigenvectors)) => {
//         println!("Eigenvalues: {}", eigenvalues);
//         println!("Eigenvectors: {}", eigenvectors);
//     }
//     Err(e) => println!("Error: {e}"),
// }
//
// ---[Singular Value Decomposition]---
// match a.svd(true, true) {
//     Ok((u, sigma, vt)) => {
//         println!("U (left singular vectors): {:?}", u.unwrap());
//         println!("Sigma (singular values): {:?}", sigma);
//         println!("Vt (right singular vectors): {:?}", vt.unwrap());
//     }
//     Err(e) => println!("Error: {e}"),
// }
//
// ---[Solving Linear Systems: ax = b]---
// // Example: 3x0 + 2x1 = 1
// //          2x0 - 2x1 = -2
// let a: Array2<f64> = array![[3., 2.], [2., -2.]];
// let b: Array1<f64> = array![1., -2.];
// let x = a.solve_into(b).unwrap();
// println!("Solution x: {:?}", x);  // [-0.2, 0.8]

// ---[Additional tips & gotchas]-------------------------------------------
//
// - {} (Display) only works on 1D/2D numeric arrays in some cases; {:?}
//   (Debug) always works and is the safer default when printing arrays.
//
// - mapv() ALWAYS returns an array the same shape as the input -- it maps
//   element-wise, 1:1. iter().filter() always drops elements, so it can
//   never preserve shape. These two facts are why "filter but keep shape"
//   isn't a thing: pick one tool based on which property you need.
//
// - Sentinel values (like -1 or 0) for "non-matching" positions can be
//   ambiguous if that value is also a valid data point. Prefer
//   mapv(|s| if cond { Some(s) } else { None }) -> Array<Option<T>> when
//   "no match" needs to be unambiguous.
//
// - .t() returns a VIEW (zero-copy, just changes strides/layout metadata).
//   swap_axes() also operates on strides, not data. Neither allocates.
//   flatten() and reshape() on owned arrays DO allocate new memory.
//
// - reshape() only works on ArcArray (rcarr1/rcarr2). For owned Array,
//   use into_shape_with_order((rows, cols)).unwrap() instead.
//
// - .f() on a shape tuple (e.g. (2, 4).f()) switches to column-major
//   (Fortran) memory order. Default without .f() is row-major (C order).
//   This matters for interop with BLAS/LAPACK via ndarray-linalg, which
//   sometimes requires Fortran-order arrays for certain operations.
//
// - as_slice() / as_slice_mut() return Option<&[T]> / Option<&mut [T]>,
//   returning None if the array isn't contiguous in memory (e.g. after
//   certain slicing/transposing operations). Freshly created arrays from
//   array![] or Array::random() are always contiguous, so unwrap() is safe
//   there, but it's not safe to assume this holds after arbitrary slicing.
//
// - rand crate API churn to watch for across versions:
//     thread_rng()         -> rng()                  (renamed in rand 0.9)
//     rand::distributions  -> rand::distr             (renamed)
//     Uniform::new() -> Dist        -> Result<Dist, Error> (now fallible)
//   Run `cargo tree | grep rand` to check exactly which versions are
//   pinned in Cargo.lock when something from a tutorial doesn't compile.
//
// - For reproducibility (debugging, tests, benchmarking, ML experiments),
//   always use a seeded RNG (StdRng::seed_from_u64(N)) rather than an
//   OS-entropy RNG. Same seed -> same sequence -> same output, every run,
//   every machine.
//
// - To shuffle two arrays in sync (e.g. features and labels, so row i
//   stays matched across both after shuffling), don't shuffle each array
//   independently. Instead, shuffle a Vec<usize> of indices once, then
//   apply it to both arrays via .select(Axis(0), &indices).
//
// - select(Axis(0), &indices) (e.g. for row shuffling/sampling) always
//   allocates a new array -- it does not mutate in place like
//   as_slice_mut().shuffle() does.
//
// - rand's choice-style sampling (NumPy's rng.choice() equivalent) lives
//   in rand::seq, not in ndarray:
//     .choose(&mut rng)               -> single random element
//     .choose_multiple(&mut rng, k)   -> k distinct elements, no repeats
//     looped .choose() k times        -> k elements, with repeats allowed
//     rand_distr::WeightedIndex       -> weighted sampling by index
//   All of these operate on slices, so as_slice()/as_slice_mut() is the
//   usual bridge from an ndarray Array into rand's sampling utilities.
