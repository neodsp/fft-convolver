//! `TwoStageFFTConvolver`: the cheapest overall, at the price of an uneven cost
//! per call.
//!
//! Good for:
//! - offline rendering, where only the total time matters and this is the
//!   fastest correct option
//! - swapping the impulse response at runtime, since `set_response` is
//!   real-time safe and `ThreadedFFTConvolver` has no equivalent
//! - anywhere you cannot or would rather not spawn a thread
//! - work that must never degrade: the output is always exact, only the timing
//!   is uneven
//!
//! Not for: an audio callback whose CPU budget is set by its worst call. One
//! call in every `tail_block_size / buffer_size` convolves the whole tail at
//! once and costs far more than its neighbours. See the `threaded` example.
//!
//! Run with `cargo run --release --example two_stage`.

use std::time::Instant;

use fft_convolver::TwoStageFFTConvolver;

const BUFFER_SIZE: usize = 256;
const SAMPLE_RATE: f64 = 48_000.0;
const IR_SIZE: usize = 131_072;
const RENDER_SECONDS: f64 = 10.0;

/// Stands in for an impulse response you would normally load from a file.
fn impulse_response(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32;
            (t * 0.013).sin() * 0.5 / (1.0 + t * 0.0001)
        })
        .collect()
}

fn main() {
    let ir = impulse_response(IR_SIZE);

    // Allocates. init_default picks the tail block size for you; pass your own
    // to init() to trade a lower peak cost for a higher average one.
    let mut convolver = TwoStageFFTConvolver::default();
    convolver
        .init_default(BUFFER_SIZE, &ir)
        .expect("block size is not zero");

    let total_samples = (RENDER_SECONDS * SAMPLE_RATE) as usize;
    let signal: Vec<f32> = (0..total_samples)
        .map(|i| (i as f32 * 0.017).sin() * 0.25)
        .collect();
    let mut rendered = vec![0.0_f32; total_samples];

    // No thread and no deadline, so this runs as fast as the machine allows.
    // That is the reason to prefer this convolver offline: the threaded one
    // would have to be synchronised with its worker to stay correct, which
    // costs more than it saves when there is no real-time stream to keep up
    // with.
    let start = Instant::now();
    for (input, output) in signal
        .chunks(BUFFER_SIZE)
        .zip(rendered.chunks_mut(BUFFER_SIZE))
    {
        convolver
            .process(input, output)
            .expect("input and output have the same length");
    }
    let elapsed = start.elapsed();

    let peak = rendered.iter().fold(0.0_f32, |peak, s| peak.max(s.abs()));
    println!(
        "rendered {RENDER_SECONDS} s of audio in {:.0} ms ({:.0}x real time), peak {peak:.3}",
        elapsed.as_secs_f64() * 1_000.0,
        RENDER_SECONDS / elapsed.as_secs_f64(),
    );

    // Both of these are real-time safe, so unlike with the threaded convolver
    // they can be called straight from an audio callback. The new impulse
    // response has to fit in the one init() was given.
    convolver
        .set_response(&impulse_response(IR_SIZE / 2))
        .expect("not longer than the original");
    convolver.reset();

    println!("impulse response swapped and state cleared, without allocating");
}
