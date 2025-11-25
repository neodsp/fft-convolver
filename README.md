# fft-convolver

Fast, real-time safe FFT-based convolution for audio processing in Rust.

Port of [HiFi-LoFi/FFTConvolver](https://github.com/HiFi-LoFi/FFTConvolver) to pure Rust.

## Features

- **Real-time safe**: No allocations, locks, or unpredictable operations during audio processing
- **Highly efficient**: Partitioned FFT convolution algorithm with uniform block sizes
- **Zero latency**: Output is sample-aligned with input (excluding processing time)
- **Flexible**: Handles arbitrary input/output buffer sizes through internal buffering
- **Generic**: Works with `f32` and `f64` floating-point types

Perfect for real-time audio applications like convolution reverbs, cabinet simulators, and other impulse response-based effects.

## How it Works

The convolver uses a partitioned FFT convolution algorithm that divides the impulse response into uniform blocks. This approach provides:

- Consistent processing time regardless of buffer size
- Efficient computation through FFT
- Low latency suitable for real-time audio

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

## Real-Time Safety

The following operations are real-time safe (no allocations):
- `process()` - Audio processing
- `set_response()` - Updating impulse response
- `reset()` - Clearing internal state

The following operations are NOT real-time safe (perform allocations):
- `init()` - Initial setup

## License

Licensed under the MIT license.
