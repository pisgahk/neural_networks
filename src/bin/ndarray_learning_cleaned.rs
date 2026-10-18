use ndarray::prelude::*;
use ndarray_rand::{
    RandomExt,
    rand::{SeedableRng, rngs::StdRng, seq::SliceRandom},
    rand_distr::Uniform,
};

fn main() {
    // ---[Array Basics & Inspection]------------------------------------------

    let a = array![
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.],
        [13., 14., 15., 16.]
    ];

    println!("a is: {:?}", a);
    println!("ndim: {}", a.ndim()); // number of dimensions
    println!("len: {}", a.len()); // total number of elements
    println!("shape: {:?}", a.shape());
    println!("is_empty: {}", a.is_empty());

    // ---[Slicing]--------------------------------------------------------

    let basic_slice = a.slice(s![0..=1, 1..=2]); // rows 0-1, cols 1-2
    let stepped_slice = a.slice(s![0..2, 2..=3; 1]); // slicing with a step

    println!("basic_slice: {}", basic_slice);
    println!("stepped_slice: {}", stepped_slice);

    // ---[Scalar Arithmetic]-----------------------------------------------

    let line_one = a.slice(s![0, ..3]);

    println!("Scalar addition: {}", &line_one + 1.0);
    println!("Scalar subtraction: {}", &line_one - 3.0);
    println!("Scalar multiplication: {}", &line_one * 3.0);
    println!("Scalar division: {}", &line_one / 4.0);

    // ---[Comparison Arithmetic / Filtering]--------------------------------
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
    // Rules for broadcasting (same as NumPy):
    //   - Dimensions are identical in size, OR
    //   - One of the dimensions is exactly 1.

    let matrix = array![[1., 2., 3.], [4., 5., 6.]];
    let row_vector = array![[10.], [20.]];

    println!("matrix shape: {:?}", matrix.shape());
    println!("row_vector shape: {:?}", row_vector.shape());

    let broadcasted_row = row_vector.broadcast((2, 3)).unwrap();
    println!("Broadcasted addition: {}", &matrix + &broadcasted_row);

    // Example: building a multiplication table via broadcasting
    let side_one = array![[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]];
    let side_two = array![[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.], [10.]];

    println!("side_one shape: {:?}", side_one.shape());
    println!("side_two shape: {:?}", side_two.shape());
    println!("Multiplication table:\n{}", &side_one * &side_two);

    // ---[Aggregate Functions]-----------------------------------------------

    let array_one = array![[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]];

    // Axis(0) aggregates down columns, Axis(1) aggregates across rows.
    // (sum, product, mean all follow the same axis convention)
    println!("Sum along Axis(1): {}", array_one.sum_axis(Axis(1)));

    // ddof: 0 for population variance/stdev, 1 for sample variance/stdev
    println!("Variance (population): {}", array_one.var(0.));
    println!("Std dev (population): {}", array_one.std(0.));

    // ---[Random Number Generation]-------------------------------------------
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

    // ---[Shuffling]-----------------------------------------------------
    // ndarray has no shuffle of its own either -- drop down to a mutable
    // slice via as_slice_mut() and use rand::seq::SliceRandom::shuffle().
    // as_slice_mut() returns None for non-contiguous arrays, hence unwrap().

    let mut shuffle_rng = StdRng::try_from_os_rng().unwrap();
    let mut shuffle_array = Array2::<f32>::random((4, 7), &distribution);
    shuffle_array
        .as_slice_mut()
        .unwrap()
        .shuffle(&mut shuffle_rng);
    println!("Shuffled array: {:#?}", shuffle_array);
}

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
//     .choose(&mut rng)                  -> single random element
//     .choose_multiple(&mut rng, k)       -> k distinct elements, no repeats
//     looped .choose() k times            -> k elements, with repeats allowed
//     rand_distr::WeightedIndex           -> weighted sampling by index
//   All of these operate on slices, so as_slice()/as_slice_mut() is the
//   usual bridge from an ndarray Array into rand's sampling utilities.
