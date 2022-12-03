# fft-convolver

Port of [HiFi-LoFi/FFTConvolver](https://github.com/HiFi-LoFi/FFTConvolver) to pure rust.

- Highly efficient convolution of audio data (e.g. for usage in real-time convolution reverbs etc.).
- Partitioned convolution algorithm (using uniform block sizes).

## Example
```Rust
use fft_convolver::FFTConvolver;

let mut impulse_response = vec![0_f32; 100];
impulse_response[0] = 1.;

let mut convolver = FFTConvolver::default();
convolver.init(16, &impulse_response);

let input = vec![0_f32; 16];
let mut output = vec![0_f32; 16];

convolver.process(&input, &mut output);
```