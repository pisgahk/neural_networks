//! A line-by-line learning companion to `gpt_dev.ipynb`.
//!
//! The headings below intentionally follow the notebook's order. Rust needs
//! structs and functions to be declared before `main`, so the declarations
//! are above the executable cells, while `main` reads like the notebook.

use std::{collections::BTreeMap, fs, path::Path};

use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
    },
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Int, Tensor, TensorData, activation},
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

type Backend = Autodiff<NdArray>;

// Cell 37: hyperparameters.
const BATCH_SIZE: usize = 16; // how many independent sequences run in parallel
const BLOCK_SIZE: usize = 32; // maximum context length
const MAX_ITERS: usize = 5_000;
const EVAL_INTERVAL: usize = 100;
const EVAL_ITERS: usize = 200;
const LEARNING_RATE: f64 = 1e-3;
const N_EMBD: usize = 64;
const N_HEAD: usize = 4;
const N_LAYER: usize = 4;
const DROPOUT: f64 = 0.0;
const WEIGHTS_FILE: &str = "gpt_dev_weights.bin";

#[derive(Clone)]
struct Vocabulary {
    stoi: BTreeMap<char, usize>,
    itos: Vec<char>,
}

impl Vocabulary {
    // Cells 5-6: sorted unique characters and stoi/itos mappings.
    fn new(text: &str) -> Self {
        let mut itos: Vec<char> = text.chars().collect();
        itos.sort_unstable();
        itos.dedup();
        let stoi = itos.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Self { stoi, itos }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| self.stoi[&c]).collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter().map(|&i| self.itos[i]).collect()
    }
}

fn tensor_from_tokens<B: burn::tensor::backend::Backend>(
    tokens: &[usize],
    shape: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    // Cell 7: torch.tensor(..., dtype=torch.long), expressed as a Burn tensor.
    let values: Vec<i64> = tokens.iter().map(|&token| token as i64).collect();
    Tensor::from_data(TensorData::new(values, shape), device)
}

#[derive(Module, Debug)]
struct Head<B: burn::tensor::backend::Backend> {
    key: Linear<B>,
    query: Linear<B>,
    value: Linear<B>,
    dropout: Dropout,
    head_size: usize,
}

impl<B: burn::tensor::backend::Backend> Head<B> {
    fn new(head_size: usize, device: &B::Device) -> Self {
        Self {
            key: LinearConfig::new(N_EMBD, head_size)
                .with_bias(false)
                .init(device),
            query: LinearConfig::new(N_EMBD, head_size)
                .with_bias(false)
                .init(device),
            value: LinearConfig::new(N_EMBD, head_size)
                .with_bias(false)
                .init(device),
            dropout: DropoutConfig::new(DROPOUT).init(),
            head_size,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Cells 23 and 37: q @ k.T, causal mask, softmax, then weighted values.
        let [_batch, time, _channels] = x.dims();
        let k = self.key.forward(x.clone());
        let q = self.query.forward(x.clone());
        let wei = q
            .matmul(k.transpose())
            .mul_scalar((self.head_size as f64).sqrt().recip())
            .mask_fill(
                Tensor::<B, 2, burn::tensor::Bool>::tril_mask([time, time], 0, &x.device())
                    .unsqueeze::<3>(),
                f32::NEG_INFINITY,
            );
        let wei = activation::softmax(wei, 2);
        self.dropout.forward(wei).matmul(self.value.forward(x))
    }
}

#[derive(Module, Debug)]
struct MultiHeadAttention<B: burn::tensor::backend::Backend> {
    heads: Vec<Head<B>>,
    proj: Linear<B>,
    dropout: Dropout,
}

impl<B: burn::tensor::backend::Backend> MultiHeadAttention<B> {
    fn new(num_heads: usize, head_size: usize, device: &B::Device) -> Self {
        Self {
            heads: (0..num_heads)
                .map(|_| Head::new(head_size, device))
                .collect(),
            proj: LinearConfig::new(num_heads * head_size, N_EMBD).init(device),
            dropout: DropoutConfig::new(DROPOUT).init(),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Cell 37: torch.cat([h(x) for h in self.heads], dim=-1).
        let outputs = self
            .heads
            .iter()
            .map(|head| head.forward(x.clone()))
            .collect();
        self.dropout
            .forward(self.proj.forward(Tensor::cat(outputs, 2)))
    }
}

#[derive(Module, Debug)]
struct FeedFoward<B: burn::tensor::backend::Backend> {
    first: Linear<B>,
    second: Linear<B>,
    dropout: Dropout,
}

impl<B: burn::tensor::backend::Backend> FeedFoward<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            first: LinearConfig::new(N_EMBD, 4 * N_EMBD).init(device),
            second: LinearConfig::new(4 * N_EMBD, N_EMBD).init(device),
            dropout: DropoutConfig::new(DROPOUT).init(),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Cell 37: Linear -> ReLU -> Linear -> Dropout.
        self.dropout
            .forward(self.second.forward(activation::relu(self.first.forward(x))))
    }
}

#[derive(Module, Debug)]
struct Block<B: burn::tensor::backend::Backend> {
    sa: MultiHeadAttention<B>,
    ffwd: FeedFoward<B>,
    ln1: LayerNorm<B>,
    ln2: LayerNorm<B>,
}

impl<B: burn::tensor::backend::Backend> Block<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            sa: MultiHeadAttention::new(N_HEAD, N_EMBD / N_HEAD, device),
            ffwd: FeedFoward::new(device),
            ln1: LayerNormConfig::new(N_EMBD).init(device),
            ln2: LayerNormConfig::new(N_EMBD).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Cell 37: pre-normalized attention and feed-forward residuals.
        let x = x.clone() + self.sa.forward(self.ln1.forward(x));
        x.clone() + self.ffwd.forward(self.ln2.forward(x))
    }
}

#[derive(Module, Debug)]
struct BigramLanguageModel<B: burn::tensor::backend::Backend> {
    token_embedding_table: Embedding<B>,
    position_embedding_table: Embedding<B>,
    blocks: Vec<Block<B>>,
    ln_f: LayerNorm<B>,
    lm_head: Linear<B>,
    loss_fn: burn::nn::loss::CrossEntropyLoss<B>,
}

impl<B: burn::tensor::backend::Backend> BigramLanguageModel<B> {
    fn new(vocab_size: usize, device: &B::Device) -> Self {
        Self {
            token_embedding_table: EmbeddingConfig::new(vocab_size, N_EMBD).init(device),
            position_embedding_table: EmbeddingConfig::new(BLOCK_SIZE, N_EMBD).init(device),
            blocks: (0..N_LAYER).map(|_| Block::new(device)).collect(),
            ln_f: LayerNormConfig::new(N_EMBD).init(device),
            lm_head: LinearConfig::new(N_EMBD, vocab_size).init(device),
            loss_fn: burn::nn::loss::CrossEntropyLossConfig::new().init(device),
        }
    }

    fn forward(
        &self,
        idx: Tensor<B, 2, Int>,
        targets: Option<Tensor<B, 2, Int>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 1>>) {
        // Cells 13 and 37: embeddings -> transformer blocks -> logits/loss.
        let [_batch, time] = idx.dims();
        let positions =
            Tensor::<B, 1, Int>::arange(0..time as i64, &idx.device()).reshape([1, time]);
        let mut x = self.token_embedding_table.forward(idx)
            + self.position_embedding_table.forward(positions);
        for block in &self.blocks {
            x = block.forward(x);
        }
        let logits = self.lm_head.forward(self.ln_f.forward(x));
        let loss = targets.map(|targets| {
            let [batch, time, vocab] = logits.dims();
            self.loss_fn.forward(
                logits.clone().reshape([batch * time, vocab]),
                targets.reshape([batch * time]),
            )
        });
        (logits, loss)
    }

    fn generate(
        &self,
        mut idx: Vec<usize>,
        max_new_tokens: usize,
        vocab_size: usize,
        rng: &mut StdRng,
        device: &B::Device,
    ) -> Vec<usize> {
        // Cells 13 and 37: predict one token, sample it, append it, repeat.
        for _ in 0..max_new_tokens {
            let start = idx.len().saturating_sub(BLOCK_SIZE);
            let context = &idx[start..];
            let input = tensor_from_tokens::<B>(context, [1, context.len()], device);
            let (logits, _) = self.forward(input, None);
            let last = logits.slice([0..1, (context.len() - 1)..context.len(), 0..vocab_size]);
            let probabilities = activation::softmax(last.reshape([vocab_size]), 0)
                .into_data()
                .to_vec::<f32>()
                .expect("model probabilities must be readable");
            let sample = sample_from_probabilities(&probabilities, rng);
            idx.push(sample);
        }
        idx
    }
}

fn sample_from_probabilities(probabilities: &[f32], rng: &mut StdRng) -> usize {
    let threshold = rng.random::<f32>();
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if threshold <= cumulative {
            return index;
        }
    }
    probabilities.len() - 1
}

fn get_batch(
    split: &str,
    train_data: &[usize],
    val_data: &[usize],
    rng: &mut StdRng,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> (Tensor<Backend, 2, Int>, Tensor<Backend, 2, Int>) {
    // Cells 11 and 37: choose random offsets and create x/y shifted sequences.
    let data = if split == "train" {
        train_data
    } else {
        val_data
    };
    let mut inputs = Vec::with_capacity(BATCH_SIZE * BLOCK_SIZE);
    let mut targets = Vec::with_capacity(BATCH_SIZE * BLOCK_SIZE);
    for _ in 0..BATCH_SIZE {
        let start = rng.random_range(0..data.len() - BLOCK_SIZE);
        inputs.extend_from_slice(&data[start..start + BLOCK_SIZE]);
        targets.extend_from_slice(&data[start + 1..start + BLOCK_SIZE + 1]);
    }
    (
        tensor_from_tokens(&inputs, [BATCH_SIZE, BLOCK_SIZE], device),
        tensor_from_tokens(&targets, [BATCH_SIZE, BLOCK_SIZE], device),
    )
}

fn estimate_loss(
    model: &BigramLanguageModel<Backend>,
    train_data: &[usize],
    val_data: &[usize],
    rng: &mut StdRng,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
    eval_iters: usize,
) -> (f32, f32) {
    // Cell 37: @torch.no_grad() evaluation over train and validation splits.
    let mut means = [0.0; 2];
    for (index, split) in ["train", "val"].iter().enumerate() {
        for iteration in 0..eval_iters {
            let (x, y) = get_batch(split, train_data, val_data, rng, device);
            let (_, loss) = model.forward(x, Some(y));
            means[index] += loss
                .expect("targets produce a loss")
                .into_data()
                .to_vec::<f32>()
                .unwrap()[0];
            if eval_iters >= 25 && (iteration + 1) % (eval_iters / 4).max(1) == 0 {
                println!(
                    "  evaluating {split}: {}/{} batches",
                    iteration + 1,
                    eval_iters
                );
            }
        }
        means[index] /= eval_iters as f32;
    }
    (means[0], means[1])
}

fn main() {
    // Cell 1: in the notebook wget downloads input.txt. Rust expects the
    // downloaded file to already exist, so this executable is reproducible
    // without performing a network request.
    let text = fs::read_to_string(Path::new("input.txt"))
        .expect("input.txt is required; download tiny Shakespeare before running");

    // Cell 2: read the text.
    let chars = Vocabulary::new(&text);
    let data = chars.encode(&text);
    let split = data.len() * 9 / 10;
    let (train_data, val_data) = data.split_at(split);

    // Cells 3-7: inspect the dataset, build the vocabulary, and encode text.
    println!("\n--- Cells 3-7: dataset and vocabulary ---");
    println!("length of dataset in characters: {}", text.chars().count());
    println!(
        "first 1000 characters:\n{}",
        &text[..text.char_indices().nth(1000).map_or(text.len(), |(i, _)| i)]
    );
    println!(
        "unique characters: {}",
        chars.itos.iter().collect::<String>()
    );
    println!("vocabulary size: {}", chars.itos.len());
    println!("encode(\"hii there\"): {:?}", chars.encode("hii there"));
    println!(
        "decode(encode(\"hii there\")): {}",
        chars.decode(&chars.encode("hii there"))
    );
    println!("encoded dataset shape: [{}]", data.len());
    println!(
        "first 1000 encoded tokens: {:?}",
        &data[..data.len().min(1000)]
    );

    // Cells 8-10: train/validation split and a single context/target example.
    println!("\n--- Cells 8-10: train/validation split and context ---");
    let context = &train_data[..BLOCK_SIZE];
    let target = &train_data[1..BLOCK_SIZE + 1];
    for time in 0..BLOCK_SIZE.min(4) {
        println!(
            "when input is {:?} the target is {}",
            &context[..time + 1],
            target[time]
        );
    }

    // Cells 11-12: get a batch and inspect it.
    println!("\n--- Cells 11-12: batch loading ---");
    let batch_rng = &mut StdRng::seed_from_u64(1337);
    let device = Default::default();
    let (xb, yb) = get_batch("train", train_data, val_data, batch_rng, &device);
    println!("initial batch inputs shape: {:?}", xb.dims());
    println!("initial batch targets shape: {:?}", yb.dims());
    println!("initial batch inputs:\n{:?}", xb.to_data());
    println!("initial batch targets:\n{:?}", yb.to_data());

    // Cells 17-34: self-attention mathematics.
    //
    // The notebook first demonstrates weighted aggregation with a lower
    // triangular matrix, then replaces that matrix with learned key/query/
    // value projections. `Head::forward` above is the direct Burn equivalent
    // of those increasingly complete examples. The same causal mask and
    // scaled softmax are used by the final model.
    println!("\n--- Cells 17-34: self-attention ---");
    println!("self-attention is implemented in Head::forward");
    println!("layer normalization is implemented in Block::forward");

    // Cells 13-16: first construct the model, run it, optimize it, and
    // generate text. The full model below corresponds to cell 37.
    println!("\n--- Cells 13-16 and 37: model, training, generation ---");
    let mut rng = StdRng::seed_from_u64(1337);
    let mut model = BigramLanguageModel::<Backend>::new(chars.itos.len(), &device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    if Path::new(WEIGHTS_FILE).exists() {
        println!("loading existing weights from {WEIGHTS_FILE}...");
        model = model
            .load_file(WEIGHTS_FILE, &recorder, &device)
            .expect("failed to load gpt_dev_weights.bin");
    } else {
        println!("no existing weights found; starting from a fresh model");
    }
    let mut optimizer = AdamWConfig::new().with_weight_decay(1e-4).init();
    let max_iters = env_usize("GPT_MAX_ITERS", MAX_ITERS).max(1);
    let eval_interval = env_usize("GPT_EVAL_INTERVAL", EVAL_INTERVAL).max(1);
    let eval_iters = env_usize("GPT_EVAL_ITERS", EVAL_ITERS).max(1);

    println!(
        "starting training: {max_iters} iterations, evaluating every {eval_interval} steps \
         ({eval_iters} train + {eval_iters} validation batches)"
    );
    for step in 0..max_iters {
        if step % eval_interval == 0 || step == max_iters - 1 {
            println!("evaluating at step {step}...");
            let (train_loss, val_loss) =
                estimate_loss(&model, train_data, val_data, &mut rng, &device, eval_iters);
            println!("step {step}: train loss {train_loss:.4}, val loss {val_loss:.4}");
        }

        let (xb, yb) = get_batch("train", train_data, val_data, &mut rng, &device);
        let (_, loss) = model.forward(xb, Some(yb));
        let loss = loss.expect("training targets produce a loss");
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(LEARNING_RATE, model, grads);
    }

    // This writes the trained weights in a gpt_dev_weights.bin file.
    println!("saving weights to {WEIGHTS_FILE}...");
    model
        .clone()
        .save_file(WEIGHTS_FILE, &recorder)
        .expect("failed to save gpt_dev_weights.bin");

    let generated_tokens = env_usize("GPT_GENERATED_TOKENS", 2_000);
    println!("generating {generated_tokens} tokens...");
    let generated = model.generate(
        vec![0],
        generated_tokens,
        chars.itos.len(),
        &mut rng,
        &device,
    );
    println!("{}", chars.decode(&generated));
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
