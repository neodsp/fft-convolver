//! `ThreadedFFTConvolver` with a real-time priority on its worker thread.
//!
//! Good for: everything the `threaded` example is good for, on a machine busy
//! enough that an ordinary thread might not meet the deadline. Reach for this
//! when `missed_blocks()` comes back non-zero.
//!
//! Nothing is done to the worker's priority by default. The right one sits just
//! below the priority your host gave the audio callback, which this crate has
//! no way to know, and guessing too high would let the tail convolution preempt
//! the callback itself. So it is left to you.
//!
//! `init_with_setup` runs a closure on the worker thread before it starts
//! convolving, which is what this needs: on Linux and macOS a real-time
//! priority has to be requested from the thread itself, not from the outside.
//!
//! Promotion needs privileges, `RLIMIT_RTPRIO` or `CAP_SYS_NICE` on Linux, so
//! it may well fail when you run this. That is not fatal and the example
//! carries on: the worker gets a full tail block period to deliver, which an
//! ordinary thread usually meets.
//!
//! This uses the `audio_thread_priority` crate, which is a dev-dependency here.
//! Any other way of setting the priority works just as well, including your
//! host's own API, such as joining the device's workgroup on macOS.
//!
//! Run with `cargo run --release --example thread_priority`.

use std::thread;
use std::time::Duration;

use audio_thread_priority::promote_current_thread_to_real_time;
use fft_convolver::{ThreadedFFTConvolver, compute_tail_block_size};

const BUFFER_SIZE: usize = 256;
const SAMPLE_RATE: u32 = 48_000;
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

    // The same tail block size init_default would have picked.
    let tail_block_size = compute_tail_block_size(BUFFER_SIZE, ir.len());

    let mut convolver = ThreadedFFTConvolver::default();
    convolver
        .init_with_setup(BUFFER_SIZE, tail_block_size, &ir, || {
            // Runs on the worker thread, before it convolves anything.
            match promote_current_thread_to_real_time(BUFFER_SIZE as u32, SAMPLE_RATE) {
                // Keeping the handle would let you demote the thread later.
                Ok(_handle) => println!("worker thread promoted to real-time"),
                Err(error) => println!("worker stays at normal priority: {error}"),
            }
        })
        .expect("block size is not zero and the thread starts");

    let input = vec![0.25_f32; BUFFER_SIZE];
    let mut output = vec![0.0_f32; BUFFER_SIZE];
    let callback_period = Duration::from_secs_f64(BUFFER_SIZE as f64 / SAMPLE_RATE as f64);

    // Stands in for the audio callback.
    for _ in 0..CALLBACKS {
        convolver
            .process(&input, &mut output)
            .expect("input and output have the same length");
        thread::sleep(callback_period);
    }

    println!("missed block periods: {}", convolver.missed_blocks());
}
