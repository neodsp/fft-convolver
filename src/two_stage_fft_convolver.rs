// TODO:
// 1. adapt comments to rust style
// 2. test the convolver

use num::Zero;
use realfft::FftError;
use rustfft::FftNum;
use thiserror::Error;

use crate::{utilities::next_power_of_2, FftConvolver, FftConvolverError};

#[derive(Error, Debug)]
pub enum TwoStageFftConvolverError {
    #[error("tail must be smaller than head")]
    TailTooBig,
    #[error("{0}")]
    FftConvolverError(#[from] FftConvolverError),
    #[error("fft error")]
    Fft(#[from] FftError),
    #[error("fatal error")]
    Fatal,
}

pub struct TwoStageFftConvolver<F: FftNum> {
    head_block_size: usize,
    tail_block_size: usize,
    head_convolver: FftConvolver<F>,
    tail_convolver0: FftConvolver<F>,
    tail_output0: Vec<F>,
    tail_precalculated0: Vec<F>,
    tail_concolver: FftConvolver<F>,
    tail_output: Vec<F>,
    tail_precalculated: Vec<F>,
    tail_input: Vec<F>,
    tail_input_fill: usize,
    precalculated_pos: usize,
}

impl<F: FftNum> Default for TwoStageFftConvolver<F> {
    fn default() -> Self {
        Self {
            head_block_size: Default::default(),
            tail_block_size: Default::default(),
            head_convolver: Default::default(),
            tail_convolver0: Default::default(),
            tail_output0: Default::default(),
            tail_precalculated0: Default::default(),
            tail_concolver: Default::default(),
            tail_output: Default::default(),
            tail_precalculated: Default::default(),
            tail_input: Default::default(),
            tail_input_fill: Default::default(),
            precalculated_pos: Default::default(),
        }
    }
}

impl<F: FftNum + std::cmp::PartialOrd<f64>> TwoStageFftConvolver<F> {
    /**
     * @brief Initialization the convolver
     * @param headBlockSize The head block size
     * @param tailBlockSize the tail block size
     * @param ir The impulse response
     * @param irLen Length of the impulse response in samples
     * @return true: Success - false: Failed
     */
    pub fn init(
        &mut self,
        head_block_size: usize,
        tail_block_size: usize,
        impulse_response: &[F],
    ) -> Result<(), TwoStageFftConvolverError> {
        self.reset();

        if head_block_size.is_zero() || tail_block_size.is_zero() {
            return Err(TwoStageFftConvolverError::FftConvolverError(
                FftConvolverError::BlockSizeZero,
            ));
        }

        let head_block_size = 1.max(head_block_size);
        if head_block_size > tail_block_size {
            return Err(TwoStageFftConvolverError::TailTooBig);
        }

        // ignore zeros at the end of the impulse response because they only waste computation time
        let mut ir_len = impulse_response.len();
        while ir_len > 0 && impulse_response[ir_len - 1] < 1e-6 {
            ir_len -= 1;
        }

        if ir_len == 0 {
            return Ok(());
        }

        self.head_block_size = next_power_of_2(head_block_size);
        self.tail_block_size = next_power_of_2(tail_block_size);

        let head_ir_len = ir_len.min(self.tail_block_size);
        self.head_convolver
            .init(head_block_size, &impulse_response[..head_ir_len])?;

        if ir_len > self.tail_block_size {
            let conv1_ir_len = (ir_len - self.tail_block_size).min(self.tail_block_size);
            self.tail_convolver0.init(
                self.head_block_size,
                &impulse_response[self.tail_block_size..self.tail_block_size + conv1_ir_len],
            )?;
            self.tail_output0.resize(self.tail_block_size, F::zero());
            self.tail_precalculated0
                .resize(self.tail_block_size, F::zero());
        }

        if ir_len > 2 * self.tail_block_size {
            let tail_ir_len = ir_len - (2 * self.tail_block_size);
            self.tail_concolver.init(
                self.tail_block_size,
                &impulse_response[2 * self.tail_block_size..2 * self.tail_block_size + tail_ir_len],
            )?;
            self.tail_output.resize(self.tail_block_size, F::zero());
            self.tail_precalculated
                .resize(self.tail_block_size, F::zero());
        }

        if !self.tail_precalculated0.is_empty() || !self.tail_precalculated.is_empty() {
            self.tail_input.resize(self.tail_block_size, F::zero());
        }

        self.tail_input_fill = 0;
        self.precalculated_pos = 0;

        Ok(())
    }

    /**
     * @brief Convolves the the given input samples and immediately outputs the result
     * @param input The input samples
     * @param output The convolution result
     * @param len Number of input/output samples
     */
    pub fn process(
        &mut self,
        input: &[F],
        output: &mut [F],
    ) -> Result<(), TwoStageFftConvolverError> {
        if input.len() != output.len() {
            return Err(TwoStageFftConvolverError::FftConvolverError(
                FftConvolverError::BufferLengthMissmatch,
            ));
        }

        // head
        self.head_convolver.process(input, output)?;

        // tail
        if !self.tail_input.is_empty() {
            let mut processed = 0;
            while processed < output.len() {
                let remaining = output.len() - processed;
                let processing = usize::min(
                    remaining,
                    self.head_block_size - (self.tail_input_fill % self.head_block_size),
                );
                if self.tail_input_fill + processing > self.tail_block_size {
                    return Err(TwoStageFftConvolverError::Fatal);
                }

                // sum head and tail
                let sum_begin = processed;
                let sum_end = processed + processing;
                {
                    // sum: 1st tail back
                    if !self.tail_precalculated0.is_empty() {
                        let mut precalculated_pos = self.precalculated_pos;
                        for i in sum_begin..sum_end {
                            output[i] = output[i] + self.tail_precalculated0[precalculated_pos];
                            precalculated_pos += 1;
                        }
                    }

                    // sum: 2nd-nth tail block
                    if !self.tail_precalculated.is_empty() {
                        let mut precalculated_pos = self.precalculated_pos;
                        for i in sum_begin..sum_end {
                            output[i] = output[i] + self.tail_precalculated[precalculated_pos];
                            precalculated_pos += 1;
                        }
                    }

                    self.precalculated_pos += processing;
                }

                // fill input buffer for tail convolution
                self.tail_input[self.tail_input_fill..self.tail_input_fill + processing]
                    .copy_from_slice(&input[processed..processed + processing]);
                self.tail_input_fill += processing;
                if self.tail_input_fill > self.tail_block_size {
                    return Err(TwoStageFftConvolverError::Fatal);
                }

                // convolution: 1st tail block
                if !self.tail_precalculated0.is_empty()
                    && self.tail_input_fill % self.head_block_size == 0
                {
                    if self.tail_input_fill < self.head_block_size {
                        return Err(TwoStageFftConvolverError::Fatal);
                    }
                    let block_offset = self.tail_input_fill - self.head_block_size;
                    self.tail_convolver0.process(
                        &self.tail_input[block_offset..block_offset + self.head_block_size],
                        &mut self.tail_output[block_offset..block_offset + self.head_block_size],
                    )?;
                    if self.tail_input_fill == self.tail_block_size {
                        std::mem::swap(&mut self.tail_precalculated0, &mut self.tail_output0);
                    }
                }

                // convolution: 2nd-nth tail block
                if !self.tail_precalculated.is_empty()
                    && self.tail_input_fill == self.tail_block_size
                    && self.tail_input.len() == self.tail_block_size
                    && self.tail_output.len() == self.tail_block_size
                {
                    std::mem::swap(&mut self.tail_precalculated, &mut self.tail_output);
                    // TODO: this could be processed in a background thread
                    self.tail_concolver
                        .process(&self.tail_input, &mut self.tail_output)?;
                }

                if self.tail_input_fill == self.tail_block_size {
                    self.tail_input_fill = 0;
                    self.precalculated_pos = 0;
                }

                processed += processing;
            }
        }

        Ok(())
    }

    /**
     * @brief Resets the convolver and discards the set impulse response
     */
    pub fn reset(&mut self) {
        *self = Default::default();
    }
}
