//! `ThreadedFFTConvolver`: long impulse responses at a steady cost per
//! callback, with the tail on a worker thread it spawns and owns.
//!
//! Good for:
//! - an audio callback with a long impulse response, where the CPU budget is
//!   set by the worst callback rather than the average one
//! - keeping the low average cost of the two-stage split without its periodic
//!   spike, at the price of one thread
//!
//! Not for:
//! - offline rendering, where `TwoStageFFTConvolver` is faster: the worker has
//!   a deadline, so correctness would force you to wait for it
//! - swapping the impulse response at runtime, which needs `set_response` and
//!   so `TwoStageFFTConvolver`
//! - anywhere a thread cannot be spawned
//!
//! See `thread_priority` for giving the worker a priority, and `custom_thread`
//! for running it on a thread of your own.
//!
//! Run with `cargo run --release --example threaded`.

use std::thread;
use std::time::Duration;

use fft_convolver::ThreadedFFTConvolver;

const BUFFER_SIZE: usize = 256;
const SAMPLE_RATE: f64 = 48_000.0;
const IR_SIZE: usize = 131_072;
const CALLBACKS: usize = 200;

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

    // Allocates and starts the worker thread, so do it before the audio thread
    // starts. init_default picks the tail block size; pass your own to init()
    // if you would rather choose the period the worker has to deliver in.
    let mut convolver = ThreadedFFTConvolver::default();
    convolver
        .init_default(BUFFER_SIZE, &ir)
        .expect("block size is not zero and the thread starts");

    println!("tail block size: {} samples", convolver.tail_block_size());

    let input = vec![0.25_f32; BUFFER_SIZE];
    let mut output = vec![0.0_f32; BUFFER_SIZE];
    let callback_period = Duration::from_secs_f64(BUFFER_SIZE as f64 / SAMPLE_RATE);

    // Stands in for the audio callback. Nothing here allocates or waits for
    // the worker; the only syscall is the wake that hands a block over, once
    // per tail block.
    for _ in 0..CALLBACKS {
        convolver
            .process(&input, &mut output)
            .expect("input and output have the same length");

        // A device would call us at this rate. The worker needs it: its
        // deadline is one tail block period, not one callback.
        thread::sleep(callback_period);
    }

    // Zero on a healthy system. Anything else means the worker did not finish
    // in time and the tail went quiet for that block period; the usual fix is
    // to give it a real-time priority, see the thread_priority example.
    println!("missed block periods: {}", convolver.missed_blocks());

    // Dropping joins the worker thread, so drop the convolver from a normal
    // thread rather than from the audio callback.
    drop(convolver);
}
