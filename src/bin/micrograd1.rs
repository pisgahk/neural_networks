// =============================================================================
// SECTION 1 — Value: scalar autograd engine (replaces the Python Value class)
// =============================================================================

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;

/// Internal mutable state of a single scalar node in the computation graph.
struct ValueInner {
    data: f64,
    grad: f64,
    label: String,
    op: String,
    prev: Vec<Value>,
    backward_fn: Option<Box<dyn Fn()>>,
}

/// A reference-counted, interior-mutable scalar value with autograd support.
/// This is the direct equivalent of the Python `Value` class.
#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueInner>>);

impl Value {
    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------
    pub fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(ValueInner {
            data,
            grad: 0.0,
            label: String::new(),
            op: String::new(),
            prev: vec![],
            backward_fn: None,
        })))
    }

    pub fn with_label(data: f64, label: &str) -> Self {
        let v = Self::new(data);
        v.set_label(label);
        v
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------
    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn set_data(&self, d: f64) {
        self.0.borrow_mut().data = d;
    }

    pub fn set_grad(&self, g: f64) {
        self.0.borrow_mut().grad = g;
    }

    pub fn set_label(&self, l: &str) {
        self.0.borrow_mut().label = l.to_string();
    }

    pub fn label(&self) -> String {
        self.0.borrow().label.clone()
    }

    // Used so HashSet can identify nodes by pointer identity (same as Python id())
    fn ptr_id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    // ------------------------------------------------------------------
    // Arithmetic operations  (mirrors __add__, __mul__, __pow__, etc.)
    // ------------------------------------------------------------------

    /// Addition:  self + other
    pub fn add(&self, other: &Value) -> Value {
        let out = Value::new(self.data() + other.data());
        out.0.borrow_mut().op = "+".into();
        out.0.borrow_mut().prev = vec![self.clone(), other.clone()];

        let s = self.clone();
        let o = other.clone();
        let out_ref = out.clone();
        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let og = out_ref.grad();
            s.0.borrow_mut().grad += og;
            o.0.borrow_mut().grad += og;
        }));
        out
    }

    /// Multiplication: self * other
    pub fn mul(&self, other: &Value) -> Value {
        let out = Value::new(self.data() * other.data());
        out.0.borrow_mut().op = "*".into();
        out.0.borrow_mut().prev = vec![self.clone(), other.clone()];

        let s = self.clone();
        let o = other.clone();
        let out_ref = out.clone();
        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let og = out_ref.grad();
            s.0.borrow_mut().grad += o.data() * og;
            o.0.borrow_mut().grad += s.data() * og;
        }));
        out
    }

    /// Power: self ^ exp  (integer or float exponent)
    pub fn pow(&self, exp: f64) -> Value {
        let out = Value::new(self.data().powf(exp));
        out.0.borrow_mut().op = format!("**{}", exp);
        out.0.borrow_mut().prev = vec![self.clone()];

        let s = self.clone();
        let out_ref = out.clone();
        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let og = out_ref.grad();
            s.0.borrow_mut().grad += exp * s.data().powf(exp - 1.0) * og;
        }));
        out
    }

    /// Negation: -self
    pub fn neg(&self) -> Value {
        self.mul(&Value::new(-1.0))
    }

    /// Subtraction: self - other
    pub fn sub(&self, other: &Value) -> Value {
        self.add(&other.neg())
    }

    /// Division: self / other  ==  self * other^(-1)
    pub fn div(&self, other: &Value) -> Value {
        self.mul(&other.pow(-1.0))
    }

    /// Scalar addition:  self + scalar
    pub fn add_f64(&self, scalar: f64) -> Value {
        self.add(&Value::new(scalar))
    }

    /// Scalar multiplication: self * scalar
    pub fn mul_f64(&self, scalar: f64) -> Value {
        self.mul(&Value::new(scalar))
    }

    // ------------------------------------------------------------------
    // Activation functions
    // ------------------------------------------------------------------

    /// Hyperbolic tangent activation (tanh)
    pub fn tanh(&self) -> Value {
        let x = self.data();
        let t = (2.0 * x).exp();
        let t = (t - 1.0) / (t + 1.0);
        let out = Value::new(t);
        out.0.borrow_mut().op = "tanh".into();
        out.0.borrow_mut().prev = vec![self.clone()];

        let s = self.clone();
        let out_ref = out.clone();
        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let og = out_ref.grad();
            // t is captured by value because it is Copy (f64)
            s.0.borrow_mut().grad += (1.0 - t * t) * og;
        }));
        out
    }

    // ------------------------------------------------------------------
    // Backpropagation
    // ------------------------------------------------------------------

    /// Runs backprop from this node back through the full computation graph.
    /// Equivalent to Python's `Value.backward()`.
    pub fn backward(&self) {
        // 1. Build topological order
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<usize> = HashSet::new();

        fn build_topo(v: &Value, topo: &mut Vec<Value>, visited: &mut HashSet<usize>) {
            let id = v.ptr_id();
            if !visited.contains(&id) {
                visited.insert(id);
                let children = v.0.borrow().prev.clone();
                for child in &children {
                    build_topo(child, topo, visited);
                }
                topo.push(v.clone());
            }
        }

        build_topo(self, &mut topo, &mut visited);

        // 2. Seed gradient at the root
        self.set_grad(1.0);

        // 3. Walk in reverse topological order, calling each node's backward closure
        for node in topo.iter().rev() {
            let bfn = node.0.borrow_mut().backward_fn.take();
            if let Some(f) = bfn {
                f();
                // Put it back so the graph can be reused
                node.0.borrow_mut().backward_fn = Some(f);
            }
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={}, grad={})", self.data(), self.grad())
    }
}

fn main() {
    // ---------------------------------------------------------------
    // 6a. Evaluate the quadratic  f(x) = 3x² - 4x + 5  at x=3
    // ---------------------------------------------------------------
    let f = |x: f64| 3.0 * x * x - 4.0 * x + 5.0;
    println!("f(3.0) = {}", f(3.0)); // expected: 20.0

    // ---------------------------------------------------------------
    // 6b. Numerical derivative of f at x = 2/3
    // ---------------------------------------------------------------
    let h = 1e-8_f64;
    let x = 2.0 / 3.0;
    let slope = (f(x + h) - f(x)) / h;
    println!("f'(2/3) ≈ {:.6}", slope); // expected: 0.0 (minimum of f)

    // ---------------------------------------------------------------
    // 6c. Manual gradient for  y = a*b + c   w.r.t. c
    // ---------------------------------------------------------------
    let a = 2.0_f64;
    let b = -3.0_f64;
    let c = 10.0_f64;
    let y = a * b + c;
    let y1 = a * b + (c + h);
    let grad_c = (y1 - y) / h;
    println!("dy/dc = {:.6}", grad_c); // expected: 1.0

    // ---------------------------------------------------------------
    // 6d. Build the computation graph and run backprop
    //     L = g * (a + b*c)  — same graph as the Python walkthrough
    // ---------------------------------------------------------------
    let va = Value::with_label(2.0, "a");
    let vb = Value::with_label(-3.0, "b");
    let vc = Value::with_label(10.0, "c");
    let vg = Value::with_label(-2.0, "g");

    let e = vb.mul(&vc);
    e.set_label("e");
    let d = va.add(&e);
    d.set_label("d");
    let l = vg.mul(&d);
    l.set_label("L");

    l.backward();

    println!("\n--- Autograd gradients ---");
    println!("a.grad = {:.4}  (expected  -2.0)", va.grad());
    println!("b.grad = {:.4}  (expected -20.0)", vb.grad());
    println!("c.grad = {:.4}  (expected   6.0)", vc.grad());
    println!("g.grad = {:.4}  (expected -28.0)", vg.grad());

    // ---------------------------------------------------------------
    // 6e. Single neuron with tanh  (replicates the Python walkthrough)
    //     x1=2, x2=0, w1=-3, w2=1, b=6.88137…  → o ≈ 0.7071
    // ---------------------------------------------------------------
    let x1 = Value::with_label(2.0, "x1");
    let x2 = Value::with_label(0.0, "x2");
    let w1 = Value::with_label(-3.0, "w1");
    let w2 = Value::with_label(1.0, "w2");
    let bias = Value::with_label(6.881_373_587_019_543, "b");

    let x1w1 = x1.mul(&w1);
    x1w1.set_label("x1w1");
    let x2w2 = x2.mul(&w2);
    x2w2.set_label("x2w2");
    let sum_xw = x1w1.add(&x2w2);
    sum_xw.set_label("x1w1+x2w2");
    let n = sum_xw.add(&bias);
    n.set_label("n");
    let o = n.tanh();
    o.set_label("o");

    o.backward();

    println!("\n--- Single neuron (tanh) ---");
    println!("o    = {:.6}  (expected ≈ 0.707107)", o.data());
    println!("x1.grad = {:.4}  (expected -1.5)", x1.grad());
    println!("w1.grad = {:.4}  (expected  1.0)", w1.grad());
    println!("x2.grad = {:.4}  (expected  0.5)", x2.grad());
    println!("w2.grad = {:.4}  (expected  0.0)", w2.grad());
}
