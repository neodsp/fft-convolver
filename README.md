# fft-convolver

Fast, real-time safe FFT-based convolution for audio processing in Rust.

Port of [HiFi-LoFi/FFTConvolver](https://github.com/HiFi-LoFi/FFTConvolver) to pure Rust.

## Features

- **Real-time safe**: No allocations, locks, or unpredictable operations during audio processing
- **Highly efficient**: Partitioned FFT convolution algorithm with uniform and non-uniform block sizes
- **Zero latency**: Output is sample-aligned with input (excluding processing time)
- **Flexible**: Handles arbitrary input/output buffer sizes through internal buffering
- **Generic**: Works with `f32` and `f64` floating-point types

Perfect for real-time audio applications like convolution reverbs, cabinet simulators, and other impulse response-based effects.

## How it Works

Both convolvers use a partitioned FFT convolution algorithm that divides the impulse response into blocks and accumulates results via overlap-add.

**`FFTConvolver`** uses uniform block sizes, giving consistent per-block processing time and predictable latency. It is the simpler of the two and works well for short-to-medium IRs.

**`TwoStageFFTConvolver`** uses two block sizes: a small "head" block for low-latency processing of the early IR, and a large "tail" block for efficient processing of the late IR. This keeps latency low while reducing the total number of FFT operations for long IRs (see [Benchmarks](#benchmarks)).

All memory allocation happens during initialization (`init()`), making subsequent processing (`process()`) completely allocation-free and suitable for real-time audio threads.

## Usage

### Basic Example

```rust
use fft_convolver::FFTConvolver;

// Create an impulse response (e.g., a simple delay)
let mut impulse_response = vec![0.0_f32; 100];
impulse_response[0] = 0.8;  // Direct sound
impulse_response[50] = 0.3; // Echo

// Initialize the convolver
let mut convolver = FFTConvolver::default();
convolver.init(128, &impulse_response).unwrap();

// Process audio in any buffer size
let input = vec![1.0_f32; 256];
let mut output = vec![0.0_f32; 256];
convolver.process(&input, &mut output).unwrap();
```

### Updating the Impulse Response

```rust
use fft_convolver::FFTConvolver;

let mut convolver = FFTConvolver::<f32>::default();
let ir1 = vec![0.5, 0.3, 0.2, 0.1];
convolver.init(128, &ir1).unwrap();

// Update to a different impulse response (must be ≤ original length)
let ir2 = vec![0.8, 0.6, 0.4];
convolver.set_response(&ir2).unwrap();
```

### TwoStageFFTConvolver

For long IRs, use `TwoStageFFTConvolver`. `init_default` automatically computes the optimal tail block size:

```rust
use fft_convolver::TwoStageFFTConvolver;

let ir = vec![0.5_f32; 65_536];

let mut convolver = TwoStageFFTConvolver::default();
convolver.init_default(512, &ir).unwrap();

let input = vec![1.0_f32; 512];
let mut output = vec![0.0_f32; 512];
convolver.process(&input, &mut output).unwrap();
```

Or control both block sizes explicitly via `init(head_block_size, tail_block_size, &ir)`.

### Handling Stream Discontinuities

```rust
use fft_convolver::FFTConvolver;

let mut convolver = FFTConvolver::<f32>::default();
let ir = vec![0.5, 0.3, 0.2];
convolver.init(128, &ir).unwrap();

// Process some audio...
let input = vec![1.0; 256];
let mut output = vec![0.0; 256];
convolver.process(&input, &mut output).unwrap();

// Clear state when seeking or handling playback discontinuities
convolver.reset();

// Continue processing with clean state
convolver.process(&input, &mut output).unwrap();
```

## Performance Considerations

- **Block size**: Affects CPU efficiency. Larger blocks are more efficient (better FFT performance) but require more computation per block. Typical values: 64-512 samples.
- **Impulse response length**: Longer IRs require more computation. The algorithm scales well with IR length.
- **Buffer size**: Any input/output size is supported efficiently through internal buffering.

## Benchmarks

Run the benchmarks yourself with:

```sh
cargo bench
```

Results on AMD Ryzen 9900X (CachyOS, x86-64):

| IR length | `FFTConvolver` | `TwoStageFFTConvolver` | speedup |
|---|---|---|---|
| 4 096 | 3.70 µs | 5.87 µs | −1.6× (slower) |
| 16 384 | 10.93 µs | 10.03 µs | **1.1×** |
| 65 536 | 40.82 µs | 16.13 µs | **2.5×** |
| 131 072 | 81.8 µs | 18.6 µs | **4.4×** |

`FFTConvolver` is faster for short IRs. `TwoStageFFTConvolver` becomes faster from ~16k samples onward, with the advantage growing significantly at longer IR lengths.

## Real-Time Safety

The following operations are real-time safe (no allocations) on both `FFTConvolver` and `TwoStageFFTConvolver`:
- `process()` - Audio processing
- `set_response()` - Updating impulse response
- `reset()` - Clearing internal state

The following operations are NOT real-time safe (perform allocations):
- `init()` / `init_default()` - Initial setup

## License

Licensed under the MIT license.
