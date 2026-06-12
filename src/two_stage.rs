use crate::utilities::{compute_tail_block_size, next_power_of_2};
use crate::{FFTConvolver, FFTConvolverError};
use realfft::FftNum;
use rtsan_standalone::nonblocking;

/// TwoStageFFTConvolver
/// Implementation of a partitioned FFT convolution algorithm with non-uniform block sizes.
///
/// Uses two different block sizes: a small head block size for low latency and a
/// larger tail block size for efficient processing of long impulse responses.
/// Internally uses three [`FFTConvolver`] instances:
///
/// - **Head**: Processes the first portion of the IR at the small block size (zero latency)
/// - **Tail0**: Transition stage that bridges the latency gap between head and tail
/// - **Tail**: Processes the remaining IR at the larger block size (efficient)
///
/// Some notes on how to use it:
/// - After initialization with an impulse response, subsequent data portions of
///   arbitrary length can be convolved. The convolver internally can handle
///   this by using appropriate buffering.
/// - The convolver works without "latency" (except for the required
///   processing time, of course), i.e. the output always is the convolved
///   input for each processing call.
/// - The convolver is suitable for real-time processing which means that no
///   "unpredictable" operations like allocations, locking, API calls, etc. are
///   performed during processing (all necessary allocations and preparations take
///   place during initialization).
#[derive(Clone, Debug)]
pub struct TwoStageFFTConvolver<F: FftNum> {
    ir_len: usize,
    head_block_size: usize,
    tail_block_size: usize,

    head_convolver: FFTConvolver<F>,
    tail_convolver0: FFTConvolver<F>,
    tail_convolver: FFTConvolver<F>,

    tail_input: Vec<F>,
    tail_input_fill: usize,

    tail_output0: Vec<F>,
    tail_precalculated0: Vec<F>,

    tail_output: Vec<F>,
    tail_precalculated: Vec<F>,

    // Pre-allocated buffer to avoid allocations during process().
    // Used to copy tail_input before passing to tail convolvers,
    // since both are fields of self and cannot be borrowed simultaneously.
    processing_buffer: Vec<F>,

    precalculated_pos: usize,
}

impl<F: FftNum> Default for TwoStageFFTConvolver<F> {
    fn default() -> Self {
        Self {
            ir_len: Default::default(),
            head_block_size: Default::default(),
            tail_block_size: Default::default(),
            head_convolver: Default::default(),
            tail_convolver0: Default::default(),
            tail_convolver: Default::default(),
            tail_input: Default::default(),
            tail_input_fill: Default::default(),
            tail_output0: Default::default(),
            tail_precalculated0: Default::default(),
            tail_output: Default::default(),
            tail_precalculated: Default::default(),
            processing_buffer: Default::default(),
            precalculated_pos: Default::default(),
        }
    }
}

impl<F: FftNum> TwoStageFFTConvolver<F> {
    /// Initializes the two-stage convolver with an impulse response
    ///
    /// This method sets up all internal buffers and prepares the convolver for processing.
    /// The head block size determines latency and will be rounded up to the next power of 2.
    /// The tail block size determines efficiency for the tail portion and will also be
    /// rounded up to the next power of 2.
    ///
    /// Use [`compute_tail_block_size`] to calculate an
    /// optimal tail block size based on García's formula.
    ///
    /// All memory allocations happen during initialization, making subsequent processing
    /// operations real-time safe.
    ///
    /// # Arguments
    ///
    /// * `head_block_size` - Block size for the head convolver (determines latency).
    ///   Will be rounded up to the next power of 2. Must be > 0.
    /// * `tail_block_size` - Block size for the tail convolver (determines efficiency).
    ///   Will be rounded up to the next power of 2. Must be > 0.
    ///   If smaller than `head_block_size`, the two values are automatically swapped.
    /// * `impulse_response` - The impulse response to convolve with. Can be empty.
    ///
    /// # Returns
    ///
    /// Returns `BlockSizeZero` if either block size is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use fft_convolver::{TwoStageFFTConvolver, compute_tail_block_size};
    ///
    /// let ir = vec![0.5_f32; 10000];
    /// let head_block_size = 64;
    /// let tail_block_size = compute_tail_block_size(head_block_size, ir.len());
    ///
    /// let mut convolver = TwoStageFFTConvolver::default();
    /// convolver.init(head_block_size, tail_block_size, &ir).unwrap();
    /// ```
    pub fn init(
        &mut self,
        head_block_size: usize,
        tail_block_size: usize,
        impulse_response: &[F],
    ) -> Result<(), FFTConvolverError> {
        if head_block_size == 0 || tail_block_size == 0 {
            return Err(FFTConvolverError::BlockSizeZero);
        }

        *self = Self::default();

        self.head_block_size = next_power_of_2(head_block_size);
        self.tail_block_size = next_power_of_2(tail_block_size);

        if self.head_block_size > self.tail_block_size {
            std::mem::swap(&mut self.head_block_size, &mut self.tail_block_size);
        }

        self.ir_len = impulse_response.len();
        let ir_len = self.ir_len;

        if ir_len == 0 {
            return Ok(());
        }

        let head_ir_len = ir_len.min(self.tail_block_size);
        self.head_convolver
            .init(self.head_block_size, &impulse_response[..head_ir_len])?;

        if ir_len > self.tail_block_size {
            let conv1_ir_len = (ir_len - self.tail_block_size).min(self.tail_block_size);
            self.tail_convolver0.init(
                self.head_block_size,
                &impulse_response[self.tail_block_size..self.tail_block_size + conv1_ir_len],
            )?;
            self.tail_output0 = vec![F::zero(); self.tail_block_size];
            self.tail_precalculated0 = vec![F::zero(); self.tail_block_size];
        }

        if ir_len > 2 * self.tail_block_size {
            let tail_ir_len = ir_len - 2 * self.tail_block_size;
            self.tail_convolver.init(
                self.tail_block_size,
                &impulse_response[2 * self.tail_block_size..2 * self.tail_block_size + tail_ir_len],
            )?;
            self.tail_output = vec![F::zero(); self.tail_block_size];
            self.tail_precalculated = vec![F::zero(); self.tail_block_size];
        }

        if !self.tail_precalculated0.is_empty() || !self.tail_precalculated.is_empty() {
            self.tail_input = vec![F::zero(); self.tail_block_size];
            self.processing_buffer = vec![F::zero(); self.tail_block_size];
        }

        self.tail_input_fill = 0;
        self.precalculated_pos = 0;

        Ok(())
    }

    /// Initializes the two-stage convolver with an automatically computed tail block size
    ///
    /// This is a convenience method that computes the optimal tail block size using
    /// García's formula and then calls [`init`](Self::init).
    ///
    /// # Arguments
    ///
    /// * `head_block_size` - Block size for the head convolver (determines latency).
    ///   Will be rounded up to the next power of 2. Must be > 0.
    /// * `impulse_response` - The impulse response to convolve with. Can be empty.
    ///
    /// # Returns
    ///
    /// Returns `BlockSizeZero` if head_block_size is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use fft_convolver::TwoStageFFTConvolver;
    ///
    /// let ir = vec![0.5_f32; 10000];
    /// let mut convolver = TwoStageFFTConvolver::default();
    /// convolver.init_default(64, &ir).unwrap();
    /// ```
    pub fn init_default(
        &mut self,
        head_block_size: usize,
        impulse_response: &[F],
    ) -> Result<(), FFTConvolverError> {
        let tail_block_size = compute_tail_block_size(head_block_size, impulse_response.len());
        self.init(head_block_size, tail_block_size, impulse_response)
    }

    /// Updates the impulse response without reallocating buffers
    ///
    /// This method allows changing the impulse response at runtime while maintaining
    /// real-time safety by avoiding allocations. The new impulse response must not
    /// exceed the length of the original impulse response used during initialization.
    ///
    /// # Arguments
    ///
    /// * `impulse_response` - The new impulse response (must be ≤ original length)
    ///
    /// # Returns
    ///
    /// Returns `ImpulseResponseExceedsCapacity` if the new impulse response is longer
    /// than the original one.
    ///
    /// # Example
    ///
    /// ```
    /// use fft_convolver::{TwoStageFFTConvolver, compute_tail_block_size};
    ///
    /// let ir1 = vec![0.5_f32; 10000];
    /// let head_block_size = 64;
    /// let tail_block_size = compute_tail_block_size(head_block_size, ir1.len());
    ///
    /// let mut convolver = TwoStageFFTConvolver::default();
    /// convolver.init(head_block_size, tail_block_size, &ir1).unwrap();
    ///
    /// let ir2 = vec![0.8_f32; 5000];
    /// convolver.set_response(&ir2).unwrap();
    /// ```
    #[nonblocking]
    pub fn set_response(&mut self, impulse_response: &[F]) -> Result<(), FFTConvolverError> {
        if impulse_response.len() > self.ir_len {
            return Err(FFTConvolverError::ImpulseResponseExceedsCapacity);
        }

        let ir_len = impulse_response.len();

        // Head
        let head_ir_len = ir_len.min(self.tail_block_size);
        self.head_convolver
            .set_response(&impulse_response[..head_ir_len])?;

        // Tail0
        if !self.tail_precalculated0.is_empty() {
            if ir_len > self.tail_block_size {
                let conv1_ir_len = (ir_len - self.tail_block_size).min(self.tail_block_size);
                self.tail_convolver0.set_response(
                    &impulse_response[self.tail_block_size..self.tail_block_size + conv1_ir_len],
                )?;
            } else {
                self.tail_convolver0.set_response(&[])?;
            }
            self.tail_output0.fill(F::zero());
            self.tail_precalculated0.fill(F::zero());
        }

        // Tail
        if !self.tail_precalculated.is_empty() {
            if ir_len > 2 * self.tail_block_size {
                let tail_ir_len = ir_len - 2 * self.tail_block_size;
                self.tail_convolver.set_response(
                    &impulse_response
                        [2 * self.tail_block_size..2 * self.tail_block_size + tail_ir_len],
                )?;
            } else {
                self.tail_convolver.set_response(&[])?;
            }
            self.tail_output.fill(F::zero());
            self.tail_precalculated.fill(F::zero());
        }

        self.tail_input.fill(F::zero());
        self.tail_input_fill = 0;
        self.precalculated_pos = 0;

        Ok(())
    }

    /// Convolves the input samples with the impulse response and outputs the result
    ///
    /// This is a real-time safe operation that performs no allocations. Internal buffering
    /// handles arbitrary sizes and ensures the output is always properly aligned with the
    /// input (zero latency except for processing time).
    ///
    /// # Arguments
    ///
    /// * `input` - The input samples to convolve
    /// * `output` - Buffer to write the convolution result. Must have the same length as `input`.
    ///
    /// # Returns
    ///
    /// Returns `InputOutputLengthMismatch` if `input` and `output` have different lengths.
    /// Returns `Fft` error if an FFT operation fails.
    ///
    /// # Example
    ///
    /// ```
    /// use fft_convolver::{TwoStageFFTConvolver, compute_tail_block_size};
    ///
    /// let ir = vec![0.5_f32; 10000];
    /// let head_block_size = 64;
    /// let tail_block_size = compute_tail_block_size(head_block_size, ir.len());
    ///
    /// let mut convolver = TwoStageFFTConvolver::default();
    /// convolver.init(head_block_size, tail_block_size, &ir).unwrap();
    ///
    /// let input = vec![1.0_f32; 256];
    /// let mut output = vec![0.0_f32; 256];
    /// convolver.process(&input, &mut output).unwrap();
    /// ```
    #[nonblocking]
    pub fn process(&mut self, input: &[F], output: &mut [F]) -> Result<(), FFTConvolverError> {
        if input.len() != output.len() {
            return Err(FFTConvolverError::InputOutputLengthMismatch);
        }
        self.head_convolver.process(input, output)?;

        if self.tail_input.is_empty() {
            return Ok(());
        }

        let len = input.len();
        let mut processed = 0;

        while processed < len {
            let remaining = len - processed;
            let processing =
                remaining.min(self.head_block_size - (self.tail_input_fill % self.head_block_size));

            let sum_begin = processed;
            let sum_end = processed + processing;

            // Add precalculated tail0 output
            if !self.tail_precalculated0.is_empty() {
                let precalc = &self.tail_precalculated0;
                let mut pos = self.precalculated_pos;
                #[allow(clippy::explicit_counter_loop)]
                for sample in &mut output[sum_begin..sum_end] {
                    *sample = *sample + precalc[pos];
                    pos += 1;
                }
            }

            // Add precalculated tail output
            if !self.tail_precalculated.is_empty() {
                let precalc = &self.tail_precalculated;
                let mut pos = self.precalculated_pos;
                #[allow(clippy::explicit_counter_loop)]
                for sample in &mut output[sum_begin..sum_end] {
                    *sample = *sample + precalc[pos];
                    pos += 1;
                }
            }

            self.precalculated_pos += processing;

            // Buffer input for tail processing
            self.tail_input[self.tail_input_fill..self.tail_input_fill + processing]
                .copy_from_slice(&input[processed..processed + processing]);
            self.tail_input_fill += processing;

            // Process tail0 incrementally (every head_block_size samples)
            if !self.tail_precalculated0.is_empty()
                && self.tail_input_fill.is_multiple_of(self.head_block_size)
            {
                let block_offset = self.tail_input_fill - self.head_block_size;
                self.processing_buffer[..self.head_block_size].copy_from_slice(
                    &self.tail_input[block_offset..block_offset + self.head_block_size],
                );
                self.tail_convolver0.process(
                    &self.processing_buffer[..self.head_block_size],
                    &mut self.tail_output0[block_offset..block_offset + self.head_block_size],
                )?;
                if self.tail_input_fill == self.tail_block_size {
                    std::mem::swap(&mut self.tail_precalculated0, &mut self.tail_output0);
                }
            }

            // Process tail (when full tail block accumulated)
            if !self.tail_precalculated.is_empty() && self.tail_input_fill == self.tail_block_size {
                std::mem::swap(&mut self.tail_precalculated, &mut self.tail_output);
                self.processing_buffer.copy_from_slice(&self.tail_input);
                self.tail_convolver
                    .process(&self.processing_buffer, &mut self.tail_output)?;
            }

            if self.tail_input_fill == self.tail_block_size {
                self.tail_input_fill = 0;
                self.precalculated_pos = 0;
            }

            processed += processing;
        }

        Ok(())
    }

    /// Clears the internal processing state while preserving the impulse response
    ///
    /// This real-time safe operation resets all internal buffers that store the
    /// convolution state, effectively removing any "history" or "tail" from previous
    /// processing. The impulse response configuration remains intact, so processing
    /// can continue immediately.
    ///
    /// This is useful when handling stream discontinuities such as:
    /// - Seeking in audio playback
    /// - Pause/resume operations with large time gaps
    /// - Switching between different audio sources
    ///
    /// After calling `reset()`, the next `process()` call will produce output as if
    /// the convolver had just been initialized.
    ///
    /// # Example
    ///
    /// ```
    /// use fft_convolver::{TwoStageFFTConvolver, compute_tail_block_size};
    ///
    /// let ir = vec![0.5_f32; 10000];
    /// let head_block_size = 64;
    /// let tail_block_size = compute_tail_block_size(head_block_size, ir.len());
    ///
    /// let mut convolver = TwoStageFFTConvolver::default();
    /// convolver.init(head_block_size, tail_block_size, &ir).unwrap();
    ///
    /// let input = vec![1.0_f32; 256];
    /// let mut output = vec![0.0_f32; 256];
    /// convolver.process(&input, &mut output).unwrap();
    ///
    /// convolver.reset();
    /// convolver.process(&input, &mut output).unwrap();
    /// ```
    #[nonblocking]
    pub fn reset(&mut self) {
        self.head_convolver.reset();
        self.tail_convolver0.reset();
        self.tail_convolver.reset();

        self.tail_input.fill(F::zero());
        self.tail_input_fill = 0;

        self.tail_output0.fill(F::zero());
        self.tail_precalculated0.fill(F::zero());

        self.tail_output.fill(F::zero());
        self.tail_precalculated.fill(F::zero());

        self.processing_buffer.fill(F::zero());
        self.precalculated_pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use crate::{FFTConvolver, FFTConvolverError, TwoStageFFTConvolver, compute_tail_block_size};

    #[test]
    fn init_test() {
        let mut convolver = TwoStageFFTConvolver::<f32>::default();
        let ir = vec![0.5; 10000];
        convolver.init(64, 4096, &ir).unwrap();

        assert_eq!(convolver.head_block_size, 64);
        assert_eq!(convolver.tail_block_size, 4096);
        assert_eq!(convolver.ir_len, 10000);

        // Head covers ir[0..4096], tail0 covers ir[4096..8192], tail covers ir[8192..]
        assert_eq!(convolver.tail_input.len(), 4096);
        assert_eq!(convolver.tail_output0.len(), 4096);
        assert_eq!(convolver.tail_precalculated0.len(), 4096);
        assert_eq!(convolver.tail_output.len(), 4096);
        assert_eq!(convolver.tail_precalculated.len(), 4096);
    }

    #[test]
    fn init_short_ir_test() {
        // IR shorter than head block size — only head convolver active
        let mut convolver = TwoStageFFTConvolver::<f32>::default();
        let ir = vec![0.5, 0.3, 0.2, 0.1];
        convolver.init(64, 4096, &ir).unwrap();

        assert!(convolver.tail_input.is_empty());
        assert!(convolver.tail_output0.is_empty());
        assert!(convolver.tail_precalculated0.is_empty());
        assert!(convolver.tail_output.is_empty());
        assert!(convolver.tail_precalculated.is_empty());
    }

    #[test]
    fn init_medium_ir_test() {
        // IR between tail_block_size and 2*tail_block_size — head + tail0 only
        let mut convolver = TwoStageFFTConvolver::<f32>::default();
        let ir = vec![0.5; 5000];
        convolver.init(64, 4096, &ir).unwrap();

        assert_eq!(convolver.tail_input.len(), 4096);
        assert_eq!(convolver.tail_output0.len(), 4096);
        assert_eq!(convolver.tail_precalculated0.len(), 4096);
        // No tail convolver since ir_len < 2 * tail_block_size
        assert!(convolver.tail_output.is_empty());
        assert!(convolver.tail_precalculated.is_empty());
    }

    #[test]
    fn init_block_size_zero_returns_error() {
        let mut convolver = TwoStageFFTConvolver::<f32>::default();
        let ir = vec![0.5; 100];
        assert!(matches!(
            convolver.init(0, 4096, &ir),
            Err(FFTConvolverError::BlockSizeZero)
        ));
        assert!(matches!(
            convolver.init(64, 0, &ir),
            Err(FFTConvolverError::BlockSizeZero)
        ));
    }

    #[test]
    fn init_swaps_block_sizes_if_inverted() {
        let mut convolver = TwoStageFFTConvolver::<f32>::default();
        let ir = vec![0.5; 10000];
        convolver.init(4096, 64, &ir).unwrap();

        assert_eq!(convolver.head_block_size, 64);
        assert_eq!(convolver.tail_block_size, 4096);
    }

    #[test]
    fn reinit_with_shorter_ir_does_not_leave_stale_buffers() {
        let short_ir = vec![0.5_f32; 4];

        // Re-initialized convolver
        let mut reinit = TwoStageFFTConvolver::<f32>::default();
        let long_ir = vec![0.5_f32; 10000];
        reinit.init(64, 4096, &long_ir).unwrap();
        reinit.init(64, 4096, &short_ir).unwrap();

        // Tail buffers must be empty
        assert!(reinit.tail_input.is_empty());
        assert!(reinit.tail_output0.is_empty());
        assert!(reinit.tail_precalculated0.is_empty());
        assert!(reinit.tail_output.is_empty());
        assert!(reinit.tail_precalculated.is_empty());

        // Output must match a freshly initialized convolver
        let mut fresh = TwoStageFFTConvolver::<f32>::default();
        fresh.init(64, 4096, &short_ir).unwrap();

        let input = vec![1.0_f32; 256];
        let mut output_reinit = vec![0.0_f32; 256];
        let mut output_fresh = vec![0.0_f32; 256];

        reinit.process(&input, &mut output_reinit).unwrap();
        fresh.process(&input, &mut output_fresh).unwrap();

        for i in 0..output_reinit.len() {
            assert!(
                (output_reinit[i] - output_fresh[i]).abs() < 1e-5,
                "Mismatch at index {}: reinit produced {}, fresh produced {}",
                i,
                output_reinit[i],
                output_fresh[i]
            );
        }
    }

    #[test]
    fn process_test() {
        let mut convolver = TwoStageFFTConvolver::<f32>::default();
        // IR longer than 2*tail_block_size to exercise all stages
        let mut ir = vec![0.0; 300];
        ir[0] = 1.0;
        convolver.init(4, 16, &ir).unwrap();

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 8];
        convolver.process(&input, &mut output).unwrap();

        for i in 0..output.len() {
            assert!(
                (input[i] - output[i]).abs() < 1e-5,
                "Mismatch at index {}: input={}, output={}",
                i,
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn equivalence_test() {
        // Two-stage convolver should produce same output as uniform convolver
        let ir: Vec<f32> = (0..500).map(|i| 1.0 / (i as f32 + 1.0)).collect();

        let mut uniform = FFTConvolver::<f32>::default();
        uniform.init(64, &ir).unwrap();

        let tail_block_size = compute_tail_block_size(64, ir.len());
        let mut two_stage = TwoStageFFTConvolver::<f32>::default();
        two_stage.init(64, tail_block_size, &ir).unwrap();

        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut output_uniform = vec![0.0; 1024];
        let mut output_two_stage = vec![0.0; 1024];

        uniform.process(&input, &mut output_uniform).unwrap();
        two_stage.process(&input, &mut output_two_stage).unwrap();

        for i in 0..output_uniform.len() {
            assert!(
                (output_uniform[i] - output_two_stage[i]).abs() < 1e-3,
                "Mismatch at index {}: uniform={}, two_stage={}",
                i,
                output_uniform[i],
                output_two_stage[i]
            );
        }
    }

    #[test]
    fn test_zero_latency() {
        let mut convolver = TwoStageFFTConvolver::<f32>::default();
        let ir = vec![0.5, 0.3, 0.2, 0.1];
        convolver.init(4, 16, &ir).unwrap();

        let mut input = vec![0.0; 16];
        input[0] = 1.0;

        let mut output = vec![0.0; 16];
        convolver.process(&input, &mut output).unwrap();

        assert!(
            output[0].abs() > 0.0,
            "Output[0] should be non-zero, indicating zero latency. Got: {}",
            output[0]
        );
        assert!(
            (output[0] - 0.5).abs() < 1e-5,
            "output[0] should be 0.5, got {}",
            output[0]
        );
    }

    #[test]
    fn reset_test() {
        let ir: Vec<f32> = (0..500).map(|i| 1.0 / (i as f32 + 1.0)).collect();

        let mut convolver1 = TwoStageFFTConvolver::<f32>::default();
        convolver1.init(64, 256, &ir).unwrap();

        // Process some data to build up history
        let history_input: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut history_output = vec![0.0; 512];
        convolver1
            .process(&history_input, &mut history_output)
            .unwrap();

        convolver1.reset();

        let test_input = vec![1.0; 256];
        let mut output1 = vec![0.0; 256];
        convolver1.process(&test_input, &mut output1).unwrap();

        let mut convolver2 = TwoStageFFTConvolver::<f32>::default();
        convolver2.init(64, 256, &ir).unwrap();
        let mut output2 = vec![0.0; 256];
        convolver2.process(&test_input, &mut output2).unwrap();

        for i in 0..output1.len() {
            assert!(
                (output1[i] - output2[i]).abs() < 1e-5,
                "Mismatch at index {}: cleared={}, fresh={}",
                i,
                output1[i],
                output2[i]
            );
        }
    }

    #[test]
    fn reset_preserves_configuration() {
        let ir = vec![0.5; 10000];
        let mut convolver = TwoStageFFTConvolver::<f32>::default();
        convolver.init(64, 4096, &ir).unwrap();

        let ir_len = convolver.ir_len;
        let head_block_size = convolver.head_block_size;
        let tail_block_size = convolver.tail_block_size;

        let input = vec![1.0; 256];
        let mut output = vec![0.0; 256];
        convolver.process(&input, &mut output).unwrap();

        convolver.reset();

        assert_eq!(convolver.ir_len, ir_len);
        assert_eq!(convolver.head_block_size, head_block_size);
        assert_eq!(convolver.tail_block_size, tail_block_size);
    }

    #[test]
    fn set_response_equals_init() {
        let ir1 = vec![0.5; 10000];
        let ir2: Vec<f32> = (0..10000).map(|i| 1.0 / (i as f32 + 1.0)).collect();

        let mut convolver1 = TwoStageFFTConvolver::<f32>::default();
        convolver1.init(64, 4096, &ir1).unwrap();
        convolver1.set_response(&ir2).unwrap();

        let mut convolver2 = TwoStageFFTConvolver::<f32>::default();
        convolver2.init(64, 4096, &ir2).unwrap();

        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut output1 = vec![0.0; 1024];
        let mut output2 = vec![0.0; 1024];

        convolver1.process(&input, &mut output1).unwrap();
        convolver2.process(&input, &mut output2).unwrap();

        for i in 0..output1.len() {
            assert!(
                (output1[i] - output2[i]).abs() < 1e-3,
                "Mismatch at index {}: set_response={}, init={}",
                i,
                output1[i],
                output2[i]
            );
        }
    }

    #[test]
    fn set_response_with_shorter_ir() {
        let ir1 = vec![0.5; 10000];
        let ir2 = vec![0.8; 5000];

        let mut convolver1 = TwoStageFFTConvolver::<f32>::default();
        convolver1.init(64, 4096, &ir1).unwrap();
        convolver1.set_response(&ir2).unwrap();

        let mut convolver2 = TwoStageFFTConvolver::<f32>::default();
        convolver2.init(64, 4096, &ir2).unwrap();

        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut output1 = vec![0.0; 1024];
        let mut output2 = vec![0.0; 1024];

        convolver1.process(&input, &mut output1).unwrap();
        convolver2.process(&input, &mut output2).unwrap();

        for i in 0..output1.len() {
            assert!(
                (output1[i] - output2[i]).abs() < 1e-3,
                "Mismatch at index {}: set_response={}, init={}",
                i,
                output1[i],
                output2[i]
            );
        }
    }

    #[test]
    fn set_response_too_long_returns_error() {
        let ir1 = vec![0.5; 5000];
        let ir2 = vec![0.8; 10000];

        let mut convolver = TwoStageFFTConvolver::<f32>::default();
        convolver.init(64, 4096, &ir1).unwrap();

        let result = convolver.set_response(&ir2);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            FFTConvolverError::ImpulseResponseExceedsCapacity
        ));
    }

    #[test]
    fn test_varying_chunk_sizes() {
        let ir: Vec<f32> = (0..500).map(|i| 1.0 / (i as f32 + 1.0)).collect();
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();

        // Process all at once
        let mut convolver_ref = TwoStageFFTConvolver::<f32>::default();
        convolver_ref.init(64, 256, &ir).unwrap();
        let mut output_ref = vec![0.0; 1024];
        convolver_ref.process(&input, &mut output_ref).unwrap();

        // Process in varying chunk sizes
        let chunk_sizes = [1, 7, 64, 100, 3, 256, 13, 580];
        let mut convolver = TwoStageFFTConvolver::<f32>::default();
        convolver.init(64, 256, &ir).unwrap();
        let mut output = vec![0.0; 1024];
        let mut pos = 0;
        for &chunk in &chunk_sizes {
            let end = (pos + chunk).min(1024);
            if pos >= end {
                break;
            }
            convolver
                .process(&input[pos..end], &mut output[pos..end])
                .unwrap();
            pos = end;
        }

        for i in 0..output_ref.len() {
            assert!(
                (output_ref[i] - output[i]).abs() < 1e-3,
                "Mismatch at index {}: ref={}, chunked={}",
                i,
                output_ref[i],
                output[i]
            );
        }
    }

    #[test]
    fn process_mismatched_lengths_returns_error() {
        let mut convolver = TwoStageFFTConvolver::<f32>::default();
        let ir = vec![0.5; 10000];
        convolver.init(64, 4096, &ir).unwrap();

        let input = vec![1.0_f32; 64];
        let mut output = vec![0.0_f32; 128];
        let result = convolver.process(&input, &mut output);
        assert!(matches!(
            result.unwrap_err(),
            FFTConvolverError::InputOutputLengthMismatch
        ));
    }

    #[test]
    fn test_short_ir_degenerate() {
        // IR shorter than head block — should behave like plain FFTConvolver
        let ir = vec![0.5, 0.3, 0.2, 0.1];

        let mut uniform = FFTConvolver::<f32>::default();
        uniform.init(64, &ir).unwrap();

        let mut two_stage = TwoStageFFTConvolver::<f32>::default();
        two_stage.init(64, 256, &ir).unwrap();

        let input = vec![1.0; 256];
        let mut output_uniform = vec![0.0; 256];
        let mut output_two_stage = vec![0.0; 256];

        uniform.process(&input, &mut output_uniform).unwrap();
        two_stage.process(&input, &mut output_two_stage).unwrap();

        for i in 0..output_uniform.len() {
            assert!(
                (output_uniform[i] - output_two_stage[i]).abs() < 1e-5,
                "Mismatch at index {}: uniform={}, two_stage={}",
                i,
                output_uniform[i],
                output_two_stage[i]
            );
        }
    }

    #[test]
    fn test_small_head_block_size() {
        // head_block_size=1 (rounds to 1); ensures no off-by-one in buffering
        let ir = vec![0.5_f32, 0.3, 0.2, 0.1];
        let mut convolver = TwoStageFFTConvolver::<f32>::default();
        convolver.init(1, 16, &ir).unwrap();
        assert_eq!(convolver.head_block_size, 1);

        let mut input = vec![0.0_f32; 16];
        input[0] = 1.0;
        let mut output = vec![0.0_f32; 16];
        convolver.process(&input, &mut output).unwrap();

        assert!((output[0] - 0.5).abs() < 1e-5);
        assert!((output[1] - 0.3).abs() < 1e-5);
        assert!((output[2] - 0.2).abs() < 1e-5);
        assert!((output[3] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_large_ir_equivalence() {
        // Long IR — all three convolver stages active; compare against uniform convolver
        let ir_len = 16384_usize;
        let mut ir = vec![0.0_f32; ir_len];
        ir[0] = 1.0; // identity

        let mut uniform = FFTConvolver::<f32>::default();
        uniform.init(512, &ir).unwrap();

        let tail_block_size = compute_tail_block_size(512, ir_len);
        let mut two_stage = TwoStageFFTConvolver::<f32>::default();
        two_stage.init(512, tail_block_size, &ir).unwrap();

        let input: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut out_uniform = vec![0.0_f32; 2048];
        let mut out_two_stage = vec![0.0_f32; 2048];

        uniform.process(&input, &mut out_uniform).unwrap();
        two_stage.process(&input, &mut out_two_stage).unwrap();

        for i in 0..input.len() {
            assert!(
                (out_uniform[i] - out_two_stage[i]).abs() < 1e-3,
                "Mismatch at {}: uniform={}, two_stage={}",
                i,
                out_uniform[i],
                out_two_stage[i]
            );
        }
    }
}
