mod fft;
mod utilities;
use crate::fft::FFT;
use crate::utilities::{
    complex_multiply_accumulate, complex_size, copy_and_pad, next_power_of_2, sum,
};
use rustfft::num_complex::Complex;

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
#[derive(Debug)]
pub struct FFTConvolver {
    ir_len: usize,
    block_size: usize,
    seg_size: usize,
    seg_count: usize,
    active_seg_count: usize,
    fft_complex_size: usize,
    segments: Vec<Vec<Complex<f32>>>,
    segments_ir: Vec<Vec<Complex<f32>>>,
    fft_buffer: Vec<f32>,
    fft: FFT,
    pre_multiplied: Vec<Complex<f32>>,
    conv: Vec<Complex<f32>>,
    overlap: Vec<f32>,
    current: usize,
    input_buffer: Vec<f32>,
    input_buffer_fill: usize,
}

impl FFTConvolver {
    pub fn default() -> Self {
        Self {
            ir_len: 0,
            block_size: 0,
            seg_size: 0,
            seg_count: 0,
            active_seg_count: 0,
            fft_complex_size: 0,
            segments: Vec::new(),
            segments_ir: Vec::new(),
            fft_buffer: Vec::new(),
            fft: FFT::default(),
            pre_multiplied: Vec::new(),
            conv: Vec::new(),
            overlap: Vec::new(),
            current: 0,
            input_buffer: Vec::new(),
            input_buffer_fill: 0,
        }
    }

    /// Resets the convolver and discards the set impulse response
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Initializes the convolver
    ///
    /// # Arguments
    ///
    /// * `block_size` - Block size internally used by the convolver (partition size)
    ///
    /// * `impulse_response` - The impulse response
    ///
    pub fn init(&mut self, block_size: usize, impulse_response: &[f32]) -> bool {
        self.reset();

        if block_size == 0 {
            return false;
        }

        self.ir_len = impulse_response.len();

        if self.ir_len == 0 {
            return true;
        }

        self.block_size = next_power_of_2(block_size);
        self.seg_size = 2 * self.block_size;
        self.seg_count = (self.ir_len as f64 / self.block_size as f64).ceil() as usize;
        self.active_seg_count = self.seg_count;
        self.fft_complex_size = complex_size(self.seg_size);

        // FFT
        self.fft.init(self.seg_size);
        self.fft_buffer.resize(self.seg_size, 0.);

        // prepare segments
        self.segments.resize(
            self.seg_count,
            vec![Complex::new(0., 0.); self.fft_complex_size],
        );

        // prepare ir
        for i in 0..self.seg_count {
            let mut segment = vec![Complex::new(0., 0.); self.fft_complex_size];
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
            self.fft.forward(&mut self.fft_buffer, &mut segment);
            self.segments_ir.push(segment);
        }

        // prepare convolution buffers
        self.pre_multiplied
            .resize(self.fft_complex_size, Complex::new(0., 0.));
        self.conv
            .resize(self.fft_complex_size, Complex::new(0., 0.));
        self.overlap.resize(self.block_size, 0.);

        // prepare input buffer
        self.input_buffer.resize(self.block_size, 0.);
        self.input_buffer_fill = 0;

        // reset current position
        self.current = 0;

        true
    }

    /// Convolves the the given input samples and immediately outputs the result
    ///
    /// # Arguments
    ///
    /// * `input` - The input samples
    /// * `output` - The convolution result
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        if self.active_seg_count == 0 {
            output.fill(0.);
            return;
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
                .clone_from_slice(&input[processed..processed + processing]);

            // Forward FFT
            copy_and_pad(&mut self.fft_buffer, &self.input_buffer, self.block_size);
            self.fft
                .forward(&mut self.fft_buffer, &mut self.segments[self.current]);

            // complex multiplication
            if input_buffer_was_empty {
                self.pre_multiplied.fill(Complex { re: 0., im: 0. });
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
            self.conv.clone_from_slice(&self.pre_multiplied);
            complex_multiply_accumulate(
                &mut self.conv,
                &self.segments[self.current],
                &self.segments_ir[0],
            );

            // Backward FFT
            self.fft.inverse(&mut self.conv, &mut self.fft_buffer);

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
                self.input_buffer.fill(0.);
                self.input_buffer_fill = 0;
                // Save the overlap
                self.overlap
                    .clone_from_slice(&self.fft_buffer[self.block_size..self.block_size * 2]);

                // Update the current segment
                self.current = if self.current > 0 {
                    self.current - 1
                } else {
                    self.active_seg_count - 1
                };
            }
            processed += processing;
        }
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
        convolver.init(10, &ir);

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
        let mut convolver = FFTConvolver::default();
        let ir = vec![1., 0., 0., 0.];
        convolver.init(2, &ir);

        let input = vec![0., 1., 2., 3.];
        let mut output = vec![0.; 4];
        convolver.process(&input, &mut output);

        for i in 0..output.len() {
            assert_eq!(input[i], output[i]);
        }
    }
}
