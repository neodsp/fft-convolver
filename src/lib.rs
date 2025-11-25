mod fft;
mod utilities;
use crate::fft::Fft;
use crate::utilities::{
    complex_multiply_accumulate, complex_size, copy_and_pad, next_power_of_2, sum,
};
use realfft::num_complex::Complex;
use realfft::num_traits::Zero;
use realfft::{FftError, FftNum};
use rtsan_standalone::nonblocking;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FFTConvolverInitError {
    #[error("block size is not allowed to be zero")]
    BlockSizeZero(),
    #[error("fft error")]
    Fft(#[from] FftError),
}

#[derive(Error, Debug)]
pub enum FFTConvolverProcessError {
    #[error("fft error")]
    Fft(#[from] FftError),
}

/// FFTConvolver
/// Implementation of a partitioned FFT convolution algorithm with uniform block size.
///
/// Some notes on how to use it:
/// - After initialization with an impulse response, subsequent data portions of
///   arbitrary length can be convolved. The convolver internally can handle
///   this by using appropriate buffering.
/// - The convolver works without "latency" (except for the required
///   processing time, of course), i.e. the output always is the convolved
///   input for each processing call.
///
/// - The convolver is suitable for real-time processing which means that no
///   "unpredictable" operations like allocations, locking, API calls, etc. are
///   performed during processing (all necessary allocations and preparations take
///   place during initialization).
#[derive(Clone)]
pub struct FFTConvolver<F: FftNum> {
    ir_len: usize,
    block_size: usize,
    seg_size: usize,
    seg_count: usize,
    active_seg_count: usize,
    fft_complex_size: usize,
    segments: Vec<Vec<Complex<F>>>,
    segments_ir: Vec<Vec<Complex<F>>>,
    fft_buffer: Vec<F>,
    fft: Fft<F>,
    pre_multiplied: Vec<Complex<F>>,
    conv: Vec<Complex<F>>,
    overlap: Vec<F>,
    current: usize,
    input_buffer: Vec<F>,
    input_buffer_fill: usize,
}

impl<F: FftNum> Default for FFTConvolver<F> {
    fn default() -> Self {
        Self {
            ir_len: Default::default(),
            block_size: Default::default(),
            seg_size: Default::default(),
            seg_count: Default::default(),
            active_seg_count: Default::default(),
            fft_complex_size: Default::default(),
            segments: Default::default(),
            segments_ir: Default::default(),
            fft_buffer: Default::default(),
            fft: Default::default(),
            pre_multiplied: Default::default(),
            conv: Default::default(),
            overlap: Default::default(),
            current: Default::default(),
            input_buffer: Default::default(),
            input_buffer_fill: Default::default(),
        }
    }
}

impl<F: FftNum> FFTConvolver<F> {
    /// Initializes the convolver
    ///
    /// # Arguments
    ///
    /// * `block_size` - Block size internally used by the convolver (partition size)
    ///
    /// * `impulse_response` - The impulse response
    ///
    pub fn init(
        &mut self,
        block_size: usize,
        impulse_response: &[F],
    ) -> Result<(), FFTConvolverInitError> {
        if block_size == 0 {
            return Err(FFTConvolverInitError::BlockSizeZero());
        }

        self.ir_len = impulse_response.len();

        if self.ir_len == 0 {
            return Ok(());
        }

        self.block_size = next_power_of_2(block_size);
        self.seg_size = 2 * self.block_size;
        self.seg_count = (self.ir_len as f64 / self.block_size as f64).ceil() as usize;
        self.active_seg_count = self.seg_count;
        self.fft_complex_size = complex_size(self.seg_size);

        // FFT
        self.fft.init(self.seg_size);
        self.fft_buffer = vec![F::zero(); self.seg_size];

        // prepare segments
        self.segments = vec![vec![Complex::zero(); self.fft_complex_size]; self.seg_count];

        // prepare ir
        self.segments_ir.clear();
        for i in 0..self.seg_count {
            let mut segment = vec![Complex::zero(); self.fft_complex_size];
            let remaining = self.ir_len - (i * self.block_size);
            let size_copy = if remaining >= self.block_size {
                self.block_size
            } else {
                remaining
            };
            copy_and_pad(
                &mut self.fft_buffer,
                &impulse_response[i * self.block_size..],
                size_copy,
            );
            self.fft.forward(&mut self.fft_buffer, &mut segment)?;
            self.segments_ir.push(segment);
        }

        // prepare convolution buffers
        self.pre_multiplied = vec![Complex::zero(); self.fft_complex_size];
        self.conv = vec![Complex::zero(); self.fft_complex_size];
        self.overlap.resize(self.block_size, F::zero());

        // prepare input buffer
        self.input_buffer = vec![F::zero(); self.block_size];
        self.input_buffer_fill = 0;

        // reset current position
        self.current = 0;

        Ok(())
    }

    /// Convolves the the given input samples and immediately outputs the result
    ///
    /// # Arguments
    ///
    /// * `input` - The input samples
    /// * `output` - The convolution result
    #[nonblocking]
    pub fn process(
        &mut self,
        input: &[F],
        output: &mut [F],
    ) -> Result<(), FFTConvolverProcessError> {
        if self.active_seg_count == 0 {
            output.fill(F::zero());
            return Ok(());
        }

        let mut processed = 0;
        while processed < output.len() {
            let input_buffer_was_empty = self.input_buffer_fill == 0;
            let processing = std::cmp::min(
                output.len() - processed,
                self.block_size - self.input_buffer_fill,
            );

            let input_buffer_pos = self.input_buffer_fill;
            self.input_buffer[input_buffer_pos..input_buffer_pos + processing]
                .copy_from_slice(&input[processed..processed + processing]);

            // Forward FFT
            copy_and_pad(&mut self.fft_buffer, &self.input_buffer, self.block_size);
            if let Err(err) = self
                .fft
                .forward(&mut self.fft_buffer, &mut self.segments[self.current])
            {
                output.fill(F::zero());
                return Err(err.into());
            }

            // complex multiplication
            if input_buffer_was_empty {
                self.pre_multiplied.fill(Complex::zero());
                for i in 1..self.active_seg_count {
                    let index_ir = i;
                    let index_audio = (self.current + i) % self.active_seg_count;
                    complex_multiply_accumulate(
                        &mut self.pre_multiplied,
                        &self.segments_ir[index_ir],
                        &self.segments[index_audio],
                    );
                }
            }
            self.conv.copy_from_slice(&self.pre_multiplied);
            complex_multiply_accumulate(
                &mut self.conv,
                &self.segments[self.current],
                &self.segments_ir[0],
            );

            // Backward FFT
            if let Err(err) = self.fft.inverse(&mut self.conv, &mut self.fft_buffer) {
                output.fill(F::zero());
                return Err(err.into());
            }

            // Add overlap
            sum(
                &mut output[processed..processed + processing],
                &self.fft_buffer[input_buffer_pos..input_buffer_pos + processing],
                &self.overlap[input_buffer_pos..input_buffer_pos + processing],
            );

            // Input buffer full => Next block
            self.input_buffer_fill += processing;
            if self.input_buffer_fill == self.block_size {
                // Input buffer is empty again now
                self.input_buffer.fill(F::zero());
                self.input_buffer_fill = 0;
                // Save the overlap
                self.overlap
                    .copy_from_slice(&self.fft_buffer[self.block_size..self.block_size * 2]);

                // Update the current segment
                self.current = if self.current > 0 {
                    self.current - 1
                } else {
                    self.active_seg_count - 1
                };
            }
            processed += processing;
        }
        Ok(())
    }

    /// Resets the current state.
    pub fn reset(&mut self) {
        self.input_buffer.fill(F::zero());
        self.input_buffer_fill = 0;

        self.fft_buffer.fill(F::zero());
        for segment in &mut self.segments {
            segment.fill(Complex::zero());
        }

        self.conv.fill(Complex::zero());
        self.pre_multiplied.fill(Complex::zero());

        self.overlap.fill(F::zero());
        self.current = 0;
    }
}

// Tests
#[cfg(test)]
mod tests {
    use crate::FFTConvolver;

    #[test]
    fn init_test() {
        let mut convolver = FFTConvolver::default();
        let ir = vec![1., 0., 0., 0.];
        convolver.init(10, &ir).unwrap();

        assert_eq!(convolver.ir_len, 4);
        assert_eq!(convolver.block_size, 16);
        assert_eq!(convolver.seg_size, 32);
        assert_eq!(convolver.seg_count, 1);
        assert_eq!(convolver.active_seg_count, 1);
        assert_eq!(convolver.fft_complex_size, 17);

        assert_eq!(convolver.segments.len(), 1);
        assert_eq!(convolver.segments.first().unwrap().len(), 17);
        for seg in &convolver.segments {
            for num in seg {
                assert_eq!(num.re, 0.);
                assert_eq!(num.im, 0.);
            }
        }

        assert_eq!(convolver.segments_ir.len(), 1);
        assert_eq!(convolver.segments_ir.first().unwrap().len(), 17);
        for seg in &convolver.segments_ir {
            for num in seg {
                assert_eq!(num.re, 1.);
                assert_eq!(num.im, 0.);
            }
        }

        assert_eq!(convolver.fft_buffer.len(), 32);
        assert_eq!(*convolver.fft_buffer.first().unwrap(), 1.);
        for i in 1..convolver.fft_buffer.len() {
            assert_eq!(convolver.fft_buffer[i], 0.);
        }

        assert_eq!(convolver.pre_multiplied.len(), 17);
        for num in &convolver.pre_multiplied {
            assert_eq!(num.re, 0.);
            assert_eq!(num.im, 0.);
        }

        assert_eq!(convolver.conv.len(), 17);
        for num in &convolver.conv {
            assert_eq!(num.re, 0.);
            assert_eq!(num.im, 0.);
        }

        assert_eq!(convolver.overlap.len(), 16);
        for num in &convolver.overlap {
            assert_eq!(*num, 0.);
        }

        assert_eq!(convolver.input_buffer.len(), 16);
        for num in &convolver.input_buffer {
            assert_eq!(*num, 0.);
        }

        assert_eq!(convolver.input_buffer_fill, 0);
    }

    #[test]
    fn process_test() {
        let mut convolver = FFTConvolver::<f32>::default();
        let ir = vec![1., 0., 0., 0.];
        convolver.init(2, &ir).unwrap();

        let input = vec![0., 1., 2., 3.];
        let mut output = vec![0.; 4];
        convolver.process(&input, &mut output).unwrap();

        for i in 0..output.len() {
            assert_eq!(input[i], output[i]);
        }
    }

    #[test]
    fn reset_test() {
        // Create an impulse response with actual filtering characteristics
        let ir = vec![0.5, 0.3, 0.2, 0.1];
        let block_size = 4;

        // First convolver: process data, then clear, then process again
        let mut convolver1 = FFTConvolver::<f32>::default();
        convolver1.init(block_size, &ir).unwrap();

        // Process some data to build up history
        let history_input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut history_output = vec![0.0; 8];
        convolver1
            .process(&history_input, &mut history_output)
            .unwrap();

        // Clear the history
        convolver1.reset();

        // Process fresh data after clearing
        let test_input = vec![1.0, 1.0, 1.0, 1.0];
        let mut output1 = vec![0.0; 4];
        convolver1.process(&test_input, &mut output1).unwrap();

        // Second convolver: freshly initialized, process the same data
        let mut convolver2 = FFTConvolver::<f32>::default();
        convolver2.init(block_size, &ir).unwrap();
        let mut output2 = vec![0.0; 4];
        convolver2.process(&test_input, &mut output2).unwrap();

        // The outputs should be identical if clear() truly cleared all history
        for i in 0..output1.len() {
            assert!(
                (output1[i] - output2[i]).abs() < 1e-5,
                "Mismatch at index {}: cleared convolver produced {}, fresh convolver produced {}",
                i,
                output1[i],
                output2[i]
            );
        }
    }

    #[test]
    fn reset_preserves_configuration() {
        // Test that clear() preserves the convolver configuration
        let ir = vec![0.5, 0.3, 0.2, 0.1];
        let block_size = 4;

        let mut convolver = FFTConvolver::<f32>::default();
        convolver.init(block_size, &ir).unwrap();

        let ir_len = convolver.ir_len;
        let block_size_actual = convolver.block_size;
        let seg_size = convolver.seg_size;
        let seg_count = convolver.seg_count;

        // Process some data
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        convolver.process(&input, &mut output).unwrap();

        // Clear
        convolver.reset();

        // Configuration should be unchanged
        assert_eq!(convolver.ir_len, ir_len);
        assert_eq!(convolver.block_size, block_size_actual);
        assert_eq!(convolver.seg_size, seg_size);
        assert_eq!(convolver.seg_count, seg_count);
    }
}
