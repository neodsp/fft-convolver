use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

pub struct FFT {
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
}

impl std::fmt::Debug for FFT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

impl FFT {
    pub fn default() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        Self {
            fft_forward: planner.plan_fft_forward(0),
            fft_inverse: planner.plan_fft_inverse(0),
        }
    }

    pub fn init(&mut self, length: usize) {
        let mut planner = RealFftPlanner::<f32>::new();
        self.fft_forward = planner.plan_fft_forward(length);
        self.fft_inverse = planner.plan_fft_inverse(length);
    }

    pub fn forward(&self, input: &mut [f32], output: &mut [Complex<f32>]) {
        self.fft_forward.process(input, output).unwrap();
    }

    pub fn inverse(&self, input: &mut [Complex<f32>], output: &mut [f32]) {
        self.fft_inverse.process(input, output).unwrap();

        // FFT Normalization
        let len = output.len();
        for num in output.iter_mut() {
            *num = *num / len as f32;
        }
    }
}
