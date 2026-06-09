# Release Notes

## New Features

- **`TwoStageFFTConvolver`**: A new convolver using non-uniform block sizes — a small "head" block for low-latency processing of the early IR and a large "tail" block for efficient processing of the late IR. Significantly faster than `FFTConvolver` for long impulse responses (2.5× at 65k samples, 4.4× at 131k samples). Use `init_default(block_size, &ir)` to let the library compute the optimal tail block size automatically, or `init(head_block_size, tail_block_size, &ir)` for full control.

- **`Debug` derive on public types**: `FFTConvolver` and `TwoStageFFTConvolver` now derive `Debug`.

## Bug Fixes

- **Stale buffer data on re-initialization**: Calling `init()` on an already-initialized convolver now clears all internal state, preventing leftover data from a previous session from appearing in the output.

## Improvements

- **Minimum Rust version**: Updated from 1.85 to 1.87.

- **Dependency updates**: All dependencies updated to their latest versions.
