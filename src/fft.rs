use realfft::num_complex::Complex;
use realfft::{ComplexToReal, FftError, FftNum, RealFftPlanner, RealToComplex};
use rtsan_standalone::nonblocking;
use std::sync::Arc;

#[derive(Clone)]
pub struct Fft<F: FftNum> {
    fft_forward: Arc<dyn RealToComplex<F>>,
    scratch_forward: Vec<Complex<F>>,
    fft_inverse: Arc<dyn ComplexToReal<F>>,
    scratch_inverse: Vec<Complex<F>>,
}

impl<F: FftNum> Default for Fft<F> {
    fn default() -> Self {
        let mut planner = RealFftPlanner::new();
        Self {
            fft_forward: planner.plan_fft_forward(0),
            fft_inverse: planner.plan_fft_inverse(0),
            scratch_forward: Vec::new(),
            scratch_inverse: Vec::new(),
        }
    }
}

impl<F: FftNum> Fft<F> {
    pub fn init(&mut self, length: usize) {
        let mut planner = RealFftPlanner::new();
        self.fft_forward = planner.plan_fft_forward(length);
        self.fft_inverse = planner.plan_fft_inverse(length);
        self.scratch_forward = self.fft_forward.make_scratch_vec();
        self.scratch_inverse = self.fft_inverse.make_scratch_vec();
    }

    #[nonblocking]
    pub fn forward(&mut self, input: &mut [F], output: &mut [Complex<F>]) -> Result<(), FftError> {
        self.fft_forward
            .process_with_scratch(input, output, &mut self.scratch_forward)?;
        Ok(())
    }

    #[nonblocking]
    pub fn inverse(&mut self, input: &mut [Complex<F>], output: &mut [F]) -> Result<(), FftError> {
        self.fft_inverse
            .process_with_scratch(input, output, &mut self.scratch_inverse)?;

        // FFT Normalization
        let len = output.len();
        output.iter_mut().for_each(|bin| {
            *bin = *bin / F::from_usize(len).expect("usize can be converted to FftNum");
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use realfft::num_traits::Zero;

    use super::*;

    #[test]
    fn test_fft_roundtrip() {
        let mut fft = Fft::<f32>::default();
        let size = 128;
        fft.init(size);

        // Create a simple test signal
        let mut input = vec![0.0; size];
        for i in 0..size {
            input[i] = ((i * 7 + 13) % 50) as f32 / 25.0 - 1.0;
        }
        let original = input.clone();
        let mut freq = vec![Complex::<f32>::zero(); size / 2 + 1];

        // Forward then inverse should give us back the original
        fft.forward(&mut input, &mut freq).unwrap();
        fft.inverse(&mut freq, &mut input).unwrap();

        for (&result, &expected) in input.iter().zip(original.iter()) {
            assert!((result - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_fft_sine_wave_frequency() {
        let mut fft = Fft::<f32>::default();
        let size = 128;
        fft.init(size);

        // Create a pure sine wave at bin 10
        let freq_bin = 10;
        let mut input = vec![0.0; size];
        for i in 0..size {
            input[i] =
                (2.0 * std::f32::consts::PI * freq_bin as f32 * i as f32 / size as f32).sin();
        }

        let mut freq = vec![Complex::<f32>::zero(); size / 2 + 1];
        fft.forward(&mut input, &mut freq).unwrap();

        // The energy should be concentrated at bin 10
        // For a real sine wave, the FFT magnitude at the frequency bin should be size/2
        let magnitude = freq[freq_bin].norm();
        let expected_magnitude = size as f32 / 2.0;

        assert!(
            (magnitude - expected_magnitude).abs() < 0.1,
            "Expected magnitude ~{} at bin {}, got {}",
            expected_magnitude,
            freq_bin,
            magnitude
        );

        // Other bins should have much smaller magnitudes (near zero)
        for (i, f) in freq.iter().enumerate() {
            if i != freq_bin {
                assert!(
                    f.norm() < 1.0,
                    "Bin {} should be near zero, got magnitude {}",
                    i,
                    f.norm()
                );
            }
        }
    }
}
