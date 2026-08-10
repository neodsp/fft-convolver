//! Measures the per-callback timing distribution of the three convolvers.
//!
//! Good for: deciding between them on your own machine. `cargo bench` reports
//! the average cost of a `process()` call, which hides how that cost is
//! distributed; on an audio thread the deadline is set by the worst call, so
//! this reports percentiles instead.
//!
//! `TwoStageFFTConvolver` spreads its work unevenly: most calls are cheap, but
//! every `tail_block_size / buffer_size`-th call also runs the large tail FFT.
//! `ThreadedFFTConvolver` moves that call to a worker thread.
//!
//! Calls are paced like an audio callback rather than run back to back, because
//! the worker needs a block period to deliver. The stream is simulated
//! `SPEEDUP` times faster than real time to keep the run short; the worker
//! still gets `1 / SPEEDUP` of its period, which is far more than it needs. The
//! `missed` column reports whether that held: it has to stay at 0 for the
//! timings to mean anything.
//!
//! Run with `cargo run --release --example jitter`.
//!
//! Based on the benchmark by @orottier (orottier/web-audio-api-rs#620).

use std::hint::black_box;
use std::time::{Duration, Instant};

use fft_convolver::{
    FFTConvolver, ThreadedFFTConvolver, TwoStageFFTConvolver, compute_tail_block_size,
};

const BUFFER_SIZES: &[usize] = &[64, 128, 256, 512, 1024];
const IR_SIZES: &[usize] = &[4_096, 16_384, 65_536, 131_072, 262_144];
const SAMPLE_RATE: f64 = 48_000.0;
const SPEEDUP: f64 = 8.0;
const WARMUP_CALLS: usize = 200;
const MEASURED_CALLS: usize = 2_000;

#[derive(Clone, Copy)]
struct Stats {
    avg_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    max_us: f64,
}

fn make_ir(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32;
            let envelope = 1.0 / (1.0 + t * 0.0001);
            ((t * 0.013).sin() * 0.5 + (t * 0.021).cos() * 0.25) * envelope
        })
        .collect()
}

fn make_input(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32;
            (t * 0.017).sin() * 0.7 + (t * 0.031).cos() * 0.2
        })
        .collect()
}

fn summarize(times: &mut [Duration]) -> Stats {
    times.sort_unstable();

    let to_us = |duration: Duration| duration.as_nanos() as f64 / 1_000.0;
    let percentile = |p: usize, times: &[Duration]| -> f64 {
        let index = ((times.len() - 1) * p) / 100;
        to_us(times[index])
    };

    let avg_us = times.iter().map(|&duration| to_us(duration)).sum::<f64>() / times.len() as f64;

    Stats {
        avg_us,
        p50_us: percentile(50, times),
        p95_us: percentile(95, times),
        p99_us: percentile(99, times),
        max_us: to_us(times[times.len() - 1]),
    }
}

/// Busy-waits instead of sleeping, so that waking up from a sleep does not show
/// up in the measurement of the call that follows it.
fn spin_until(deadline: Instant) {
    while Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

fn measure<F>(buffer_size: usize, mut process: F) -> (Stats, f32)
where
    F: FnMut(&[f32], &mut [f32]),
{
    let input = make_input(buffer_size);
    let mut output = vec![0.0; buffer_size];
    let mut checksum = 0.0;

    let period = Duration::from_secs_f64(buffer_size as f64 / SAMPLE_RATE / SPEEDUP);
    let epoch = Instant::now();
    let mut times = Vec::with_capacity(MEASURED_CALLS);

    for call in 0..WARMUP_CALLS + MEASURED_CALLS {
        spin_until(epoch + period * call as u32);

        let start = Instant::now();
        process(black_box(&input), black_box(&mut output));
        let elapsed = start.elapsed();

        if call >= WARMUP_CALLS {
            times.push(elapsed);
        }
        checksum += output[0];
    }

    (summarize(&mut times), checksum)
}

#[allow(clippy::too_many_arguments)]
fn print_row(
    buffer_size: usize,
    ir_size: usize,
    implementation: &str,
    tail_block_size: Option<usize>,
    missed: Option<u64>,
    stats: Stats,
    checksum: f32,
) {
    let tail_block_size = tail_block_size
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string());
    let missed = missed
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string());
    let p99_over_p50 = stats.p99_us / stats.p50_us;

    println!(
        "{buffer_size},{ir_size},{implementation},{tail_block_size},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{missed},{:.3}",
        stats.avg_us,
        stats.p50_us,
        stats.p95_us,
        stats.p99_us,
        stats.max_us,
        p99_over_p50,
        checksum,
    );
}

fn main() {
    println!("# Each row measures repeated process() calls on one initialized convolver.");
    println!("# Calls are paced at {SPEEDUP}x real time against a {SAMPLE_RATE} Hz stream.");
    println!("# p99_over_p50 is the spike factor: how much worse the worst calls are");
    println!("# than a typical one. 1.0 means every callback costs the same.");
    println!("# missed: block periods in which the worker did not deliver in time.");
    println!("# Warmup calls: {WARMUP_CALLS}; measured calls: {MEASURED_CALLS}");
    println!(
        "buffer_size,ir_size,implementation,tail_block_size,avg_us,p50_us,p95_us,p99_us,max_us,p99_over_p50,missed,checksum"
    );

    for &buffer_size in BUFFER_SIZES {
        for &ir_size in IR_SIZES {
            let ir = make_ir(ir_size);
            let tail_block_size = compute_tail_block_size(buffer_size, ir_size);

            let mut fft = FFTConvolver::<f32>::default();
            fft.init(buffer_size, &ir).unwrap();
            let (stats, checksum) = measure(buffer_size, |input, output| {
                fft.process(input, output).unwrap();
            });
            print_row(
                buffer_size,
                ir_size,
                "FFTConvolver",
                None,
                None,
                stats,
                checksum,
            );

            let mut two_stage = TwoStageFFTConvolver::<f32>::default();
            two_stage.init_default(buffer_size, &ir).unwrap();
            let (stats, checksum) = measure(buffer_size, |input, output| {
                two_stage.process(input, output).unwrap();
            });
            print_row(
                buffer_size,
                ir_size,
                "TwoStageFFTConvolver",
                Some(tail_block_size),
                None,
                stats,
                checksum,
            );

            let mut threaded = ThreadedFFTConvolver::<f32>::default();
            threaded.init_default(buffer_size, &ir).unwrap();
            let (stats, checksum) = measure(buffer_size, |input, output| {
                threaded.process(input, output).unwrap();
            });
            print_row(
                buffer_size,
                ir_size,
                "ThreadedFFTConvolver",
                Some(tail_block_size),
                Some(threaded.missed_blocks()),
                stats,
                checksum,
            );
        }
    }
}
