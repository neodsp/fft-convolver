use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub struct FFT {
    pub fft_forward: Arc<dyn RealToComplex<f32>>,
    pub fft_inverse: Arc<dyn ComplexToReal<f32>>,
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
}