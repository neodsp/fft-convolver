## Unreleased

### Breaking Changes

- **`FFTConvolverError` gained a `WorkerSpawn` variant** for a worker thread that could not be started, and is now `#[non_exhaustive]`. Matches on it need a wildcard arm; in exchange, later variants will not be breaking.

### New Features

- **`ThreadedFFTConvolver`**: A two-stage convolver that runs the tail stage on a worker thread. `TwoStageFFTConvolver` computes the whole tail on the call that completes a tail block, which lowers the average cost per call but not the peak; this one keeps the head and transition stages on the audio thread and hands the tail block over instead, so the audio thread's cost stays steady. The handoff uses lock-free SPSC ring buffers and a wake based on the standard library's thread parking, never a mutex. The audio thread never waits for the worker: if a deadline is missed the tail contributes silence for that block period, `missed_blocks()` counts it, and the output is exact again afterwards. Use `sync()` for deterministic offline rendering.

- **`ThreadedFFTConvolver::init_with_setup`**: Runs a closure on the worker thread before it starts convolving, which is where a real-time priority has to be requested from on Linux and macOS. No priority is set by default, since the right value depends on what the host gave the audio callback.

- **`ThreadedFFTConvolver::split`**: Returns the convolver together with its `TailWorker` instead of spawning a thread, for callers who want to place the work themselves. `TailWorker::run` takes over a thread until the convolver is dropped, while `run_pending` does the work that is ready without blocking, so one thread can serve the tails of several convolvers.

### Chores

- **`playback-example` feature**: The `highpass_playback` example and its `audio-host`, `audio-file` and `audio-blocks` dependencies now sit behind an off-by-default feature. Nothing else in the crate used them, so `cargo test` no longer needs the ALSA and PulseAudio development headers, and most CI jobs no longer install them.

- **Examples**: One per way of using the crate, each stating at the top what it is good for: `basic`, `two_stage`, `threaded`, `thread_priority` and `custom_thread`, plus `jitter` for measuring the per-callback cost of all three convolvers.
