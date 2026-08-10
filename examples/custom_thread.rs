//! `ThreadedFFTConvolver` on a thread you own, serving several channels.
//!
//! Good for:
//! - running many convolvers without one thread each: `init()` spawns a thread
//!   per instance, which is right for one and wasteful for eight
//! - placing the work yourself, on a thread you have already configured
//! - anywhere you would rather own the thread than have one spawned for you
//!
//! Not for: a single convolver, where `init()` or `init_with_setup()` does the
//! same thing with less code. See the `threaded` and `thread_priority`
//! examples.
//!
//! `split()` hands the tail out as a [`TailWorker`] instead of spawning a
//! thread for it, and one thread can drive as many of those as you like.
//!
//! Run with `cargo run --release --example custom_thread`.

use std::thread;
use std::time::Duration;

use fft_convolver::{TailWorker, ThreadedFFTConvolver, compute_tail_block_size};

const CHANNELS: usize = 8;
const BUFFER_SIZE: usize = 256;
const SAMPLE_RATE: f64 = 48_000.0;
const IR_SIZE: usize = 131_072;
const CALLBACKS: usize = 200;

/// Convolves the tails of every channel. Owns the thread it is called on.
fn serve_tails(mut workers: Vec<TailWorker<f32>>) {
    // Set the thread up here. A real-time priority below the one your host gave
    // the audio callback belongs in this spot: on Linux and macOS it has to be
    // requested from the thread itself, and this crate sets nothing by default.

    // Hand a block over and this thread gets woken.
    for worker in &mut workers {
        worker.set_waker(thread::current());
    }

    loop {
        let mut blocks = 0;
        for worker in &mut workers {
            blocks += worker.run_pending();
        }

        // Every convolver has been dropped, so there is nothing left to serve.
        if workers.iter().all(|worker| worker.is_disconnected()) {
            return;
        }

        // Sleep until a channel hands over its next block, or is dropped.
        // The timeout is a safety net: a convolver dropped before this thread
        // registered above would have nothing to wake, and the loop would
        // otherwise sit here rather than noticing on its next pass.
        if blocks == 0 {
            thread::park_timeout(Duration::from_millis(100));
        }
    }
}

fn main() {
    let ir: Vec<f32> = (0..IR_SIZE)
        .map(|i| {
            let t = i as f32;
            (t * 0.013).sin() * 0.5 / (1.0 + t * 0.0001)
        })
        .collect();
    let tail_block_size = compute_tail_block_size(BUFFER_SIZE, ir.len());

    // One convolver per channel, one worker per convolver, one thread for all
    // of the workers.
    let mut convolvers = Vec::with_capacity(CHANNELS);
    let mut workers = Vec::with_capacity(CHANNELS);
    for _ in 0..CHANNELS {
        let (convolver, worker) = ThreadedFFTConvolver::split(BUFFER_SIZE, tail_block_size, &ir)
            .expect("valid block sizes");
        convolvers.push(convolver);
        workers.push(worker);
    }

    let tail_thread = thread::Builder::new()
        .name("convolver-tails".to_owned())
        .spawn(move || serve_tails(workers))
        .expect("could not start the tail thread");

    // Stands in for the audio callback.
    let input = vec![0.25_f32; BUFFER_SIZE];
    let mut outputs = vec![vec![0.0_f32; BUFFER_SIZE]; CHANNELS];
    let callback_period = Duration::from_secs_f64(BUFFER_SIZE as f64 / SAMPLE_RATE);

    for _ in 0..CALLBACKS {
        for (convolver, output) in convolvers.iter_mut().zip(&mut outputs) {
            convolver.process(&input, output).expect("matching lengths");
        }

        // A device would call us at this rate. The workers need it: their
        // deadline is one tail block period, not one callback.
        thread::sleep(callback_period);
    }

    let missed: u64 = convolvers
        .iter()
        .map(|convolver| convolver.missed_blocks())
        .sum();
    println!("{CHANNELS} channels, 1 worker thread, {missed} missed block periods");

    // Dropping the convolvers ends serve_tails, so the thread can be joined.
    drop(convolvers);
    tail_thread.join().expect("the tail thread panicked");
}
