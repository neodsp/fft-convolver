//! `FFTConvolver`: the plain one, with the same cost on every call.
//!
//! Good for:
//! - short to medium impulse responses, up to roughly 16k samples
//! - audio callbacks, where a steady cost per call is worth more than a low
//!   average one
//! - starting out: no thread, no tuning, nothing to get wrong
//!
//! Not for: long impulse responses, where it does several times more work than
//! the partitioned convolvers. See the `two_stage` and `threaded` examples.
//!
//! Run with `cargo run --release --example basic`.

use fft_convolver::FFTConvolver;

const BUFFER_SIZE: usize = 256;
const CALLBACKS: usize = 100;

/// Stands in for an impulse response you would normally load from a file.
fn impulse_response(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32;
            (t * 0.013).sin() * 0.5 / (1.0 + t * 0.001)
        })
        .collect()
}

fn main() {
    let ir = impulse_response(8_192);

    // Allocates, so do it before the audio thread starts. The block size is
    // rounded up to the next power of two.
    let mut convolver = FFTConvolver::<f32>::default();
    convolver
        .init(BUFFER_SIZE, &ir)
        .expect("block size is not zero");

    let input = vec![0.25_f32; BUFFER_SIZE];
    let mut output = vec![0.0_f32; BUFFER_SIZE];

    // Stands in for the audio callback. Nothing below here allocates, locks or
    // makes a syscall. Input and output only have to match each other, they do
    // not have to match the block size passed to init.
    for _ in 0..CALLBACKS {
        convolver
            .process(&input, &mut output)
            .expect("input and output have the same length");
    }

    let peak = output.iter().fold(0.0_f32, |peak, s| peak.max(s.abs()));
    println!("{CALLBACKS} callbacks of {BUFFER_SIZE} samples, last block peak {peak:.4}");

    // Swapping the impulse response is real-time safe too, as long as the new
    // one is no longer than the one init() was given.
    convolver
        .set_response(&impulse_response(4_096))
        .expect("not longer than the original");

    // And so is clearing the tail, for a seek or a stream discontinuity.
    convolver.reset();

    convolver.process(&input, &mut output).expect("same length");
    let peak = output.iter().fold(0.0_f32, |peak, s| peak.max(s.abs()));
    println!("after set_response and reset, first block peak {peak:.4}");
}
