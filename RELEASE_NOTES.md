# Release Notes

## Breaking Changes

- **`reset()` is now `nonblocking` and preserves configuration**: The `reset()` function now only clears the internal processing state (buffers, overlap, history) while preserving the impulse response configuration. Previously it reset the entire convolver. This makes it suitable for handling stream discontinuities (seeking, pause/resume) in real-time audio threads.

- **Consolidated error types**: `FFTConvolverInitError` and `FFTConvolverProcessError` have been merged into a single `FFTConvolverError` enum for simpler error handling.

- **New `set_response()` function**: Added a real-time safe method to update the impulse response without reallocating memory. The new impulse response must not exceed the length of the original one used during initialization. This enables dynamic IR changes in real-time applications.

## Improvements

- **Real-time safety verification**: The `process()`, `set_response()`, and `reset()` functions are now validated by Realtime Sanitizer to ensure they perform no allocations or blocking operations.

- **`Clone` implementation**: `FFTConvolver` now derives `Clone`, enabling usage in vectors and other collections (thanks @piedoom).

- **Enhanced test coverage**: Added comprehensive tests for `reset()`, `set_response()`, zero-latency verification, and state management.

- **Documentation overhaul**: Complete API documentation with examples, performance considerations, and real-time safety guarantees. Added module-level documentation explaining the partitioned FFT convolution algorithm.

- **Rust Edition 2024**: Updated to Rust Edition 2024.

- **Dependency updates**: All dependencies updated to their latest versions.

- **CI/CD pipeline**: Added continuous integration for automated testing and quality assurance.
