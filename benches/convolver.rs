use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fft_convolver::{FFTConvolver, TwoStageFFTConvolver};

const BLOCK_SIZE: usize = 512;
const IR_LENGTHS: &[usize] = &[4_096, 16_384, 65_536, 131_072];

fn make_ir(len: usize) -> Vec<f32> {
    (0..len).map(|i| 1.0 / (i as f32 + 1.0)).collect()
}

fn bench_convolvers(c: &mut Criterion) {
    let input = vec![0.5_f32; BLOCK_SIZE];
    let mut output = vec![0.0_f32; BLOCK_SIZE];

    let mut group = c.benchmark_group("convolver");

    for &ir_len in IR_LENGTHS {
        let ir = make_ir(ir_len);

        let mut conv = FFTConvolver::<f32>::default();
        conv.init(BLOCK_SIZE, &ir).unwrap();
        group.bench_with_input(BenchmarkId::new("FFTConvolver", ir_len), &ir_len, |b, _| {
            b.iter(|| conv.process(&input, &mut output).unwrap())
        });

        let mut conv = TwoStageFFTConvolver::<f32>::default();
        conv.init_default(BLOCK_SIZE, &ir).unwrap();
        group.bench_with_input(
            BenchmarkId::new("TwoStageFFTConvolver", ir_len),
            &ir_len,
            |b, _| b.iter(|| conv.process(&input, &mut output).unwrap()),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_convolvers);
criterion_main!(benches);
