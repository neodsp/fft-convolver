# fft-convolver

Fast, real-time safe FFT convolution for audio processing in Rust.

Partitioned FFT convolution with zero latency, no allocations or locks while processing, arbitrary input and output buffer sizes, and `f32` or `f64` samples. Perfect for convolution reverbs, cabinet simulators and other impulse response based effects.

Port of [HiFi-LoFi/FFTConvolver](https://github.com/HiFi-LoFi/FFTConvolver) to pure Rust.

## Quick Start

```rust
use fft_convolver::FFTConvolver;

// A short impulse response: direct sound plus one echo.
let mut impulse_response = vec![0.0_f32; 100];
impulse_response[0] = 0.8;
impulse_response[50] = 0.3;

// Allocates, so do this before the audio thread starts.
let mut convolver = FFTConvolver::default();
convolver.init(128, &impulse_response).unwrap();

// Real-time safe. Input and output only have to match each other, they do not
// have to match the block size above.
let input = vec![0.25_f32; 256];
let mut output = vec![0.0_f32; 256];
convolver.process(&input, &mut output).unwrap();
```

`set_response()` swaps the impulse response and `reset()` clears the tail after a seek. Both are real-time safe.

## Which convolver?

All three produce the same output. They differ in how the work is distributed.

**`FFTConvolver`** uses a uniform block size and does everything on the calling thread, which costs the same on every call. Start here. It is the right choice up to roughly 16k samples of impulse response.

**`TwoStageFFTConvolver`** adds a large tail block on top of a small head block, which is several times cheaper on average for long responses. The catch is that one call in every `tail_block_size / buffer_size` convolves the whole tail at once and costs far more than its neighbours. Choose it for offline rendering, where it is the fastest correct option, for swapping the impulse response at runtime, and anywhere you cannot spawn a thread.

**`ThreadedFFTConvolver`** splits the response the same way but hands the tail to a worker thread, so the expensive call never lands on the audio thread. Choose it for a long response in a real-time callback, where the CPU budget is set by the worst callback rather than the average one. This matters most for multi-channel signals: with `TwoStageFFTConvolver`, every channel at the same buffer size hits its tail block boundary on the same callback, so the spike scales with channel count even though the typical call does not. It has no `set_response()`, and offline it is slower than `TwoStageFFTConvolver`, because waiting for the worker costs more than it saves when there is no stream to keep up with.

## Examples

Each one starts with what it is good for and what it is not, so you can pick by reading the top of the file.

| Example | Convolver | Good for |
|---|---|---|
| `basic` | `FFTConvolver` | short to medium responses; the same cost on every call |
| `two_stage` | `TwoStageFFTConvolver` | offline rendering; swapping the response at runtime; no threads |
| `threaded` | `ThreadedFFTConvolver` | long responses on an audio thread, at a steady cost per callback |
| `thread_priority` | `ThreadedFFTConvolver` | the same, with a real-time priority on the worker |
| `custom_thread` | `ThreadedFFTConvolver` | multi-channel signals: one thread serving every channel's tail, instead of one thread each |

```sh
cargo run --release --example basic
```

Two more that are not about setting a convolver up: `jitter` measures the per-callback cost of all three so you can decide on your own hardware, and `highpass_playback` runs a highpass FIR against a real audio device. The playback one needs the ALSA and PulseAudio development headers on Linux, so it sits behind a feature:

```sh
cargo run --release --features playback-example --example highpass_playback
```

## Benchmarks

All numbers below are from one machine, an AMD Ryzen 7 7840U laptop, so the two tables are comparable with each other. Absolute times will differ on yours; run `cargo bench` and `cargo run --release --example jitter` to get numbers for your own hardware, ideally the one you plan to deploy to.

Average cost of one `process()` call, 512-sample buffer:

| IR length | `FFTConvolver` | `TwoStageFFTConvolver` | speedup |
|---|---|---|---|
| 4 096 | 5.37 µs | 8.59 µs | −1.6× (slower) |
| 16 384 | 15.38 µs | 13.89 µs | **1.1×** |
| 65 536 | 55.66 µs | 22.87 µs | **2.4×** |
| 131 072 | 113.53 µs | 26.97 µs | **4.2×** |

The average hides the spike. Median and 99th percentile of one callback, 512-sample buffer and a 262 144-sample response, in µs:

| | p50 | p99 |
|---|---|---|
| `FFTConvolver` | 236 | 363 |
| `TwoStageFFTConvolver` | 39 | 400 |
| `ThreadedFFTConvolver` | 38 | 88 |

`TwoStageFFTConvolver`'s p99 is the algorithm and reproduces run after run. What is left on the other two moves around between runs, because it is the operating system scheduling a normal-priority thread. The gap widens with channel count, since every channel reaches its tail block boundary on the same callback.

Thanks to [@orottier](https://github.com/orottier) for raising this in [web-audio-api-rs#620](https://github.com/orottier/web-audio-api-rs/issues/620).

## Real-Time Safety

`process()`, `set_response()` and `reset()` allocate nothing, take no locks and make no syscalls. `init()` and `init_default()` allocate, so call them before the audio thread starts.

`ThreadedFFTConvolver` differs in three ways: it has no `set_response()`, its `process()` performs one thread wake per tail block, and dropping it joins the worker thread, so drop it from a normal thread rather than from the callback.

Real-time safe here means no allocations, locks or syscalls. It does not mean constant time, which is what the benchmarks above are about.

These paths are annotated with [`rtsan-standalone`](https://crates.io/crates/rtsan-standalone), so violations are caught by RealtimeSanitizer:

```sh
RTSAN_ENABLE=1 cargo test --lib
```

## License

Licensed under the MIT license.
