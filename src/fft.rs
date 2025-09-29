use realfft::{ComplexToReal, FftError, RealFftPlanner, RealToComplex};
use rustfft::{num_complex::Complex, FftNum};
use std::sync::Arc;

#[derive(Clone)]
pub struct Fft<F: FftNum> {
    fft_forward: Arc<dyn RealToComplex<F>>,
    fft_inverse: Arc<dyn ComplexToReal<F>>,
}

impl<F: FftNum> std::fmt::Debug for Fft<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

impl<F: FftNum> Default for Fft<F> {
    fn default() -> Self {
        let mut planner = RealFftPlanner::new();
        Self {
            fft_forward: planner.plan_fft_forward(0),
            fft_inverse: planner.plan_fft_inverse(0),
        }
    }
}

impl<F: FftNum> Fft<F> {
    pub fn init(&mut self, length: usize) {
        let mut planner = RealFftPlanner::new();
        self.fft_forward = planner.plan_fft_forward(length);
        self.fft_inverse = planner.plan_fft_inverse(length);
    }

    pub fn forward(&self, input: &mut [F], output: &mut [Complex<F>]) -> Result<(), FftError> {
        self.fft_forward.process(input, output)?;
        Ok(())
    }

    pub fn inverse(&self, input: &mut [Complex<F>], output: &mut [F]) -> Result<(), FftError> {
        self.fft_inverse.process(input, output)?;

        // FFT Normalization
        let len = output.len();
        output.iter_mut().for_each(|bin| {
            *bin = *bin / F::from_usize(len).expect("usize can be converted to FftNum");
        });

        Ok(())
    }
}
