use criterion::{criterion_group, criterion_main, Criterion};

pub fn fft_convolver(c: &mut Criterion) {
    use fft_convolver::FftConvolver;
    let block_size = 512;

    let mut impulse_response = vec![0_f32; 44100];
    impulse_response[0] = 1.;

    let mut convolver = FftConvolver::default();
    convolver.init(block_size, &impulse_response).unwrap();

    let input = vec![0_f32; block_size];
    let mut output = vec![0_f32; block_size];

    c.bench_function("fft_convolver", |b| {
        b.iter(|| convolver.process(&input, &mut output))
    });
}

criterion_group!(benches, fft_convolver);
criterion_main!(benches);
