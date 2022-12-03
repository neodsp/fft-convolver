use rustfft::num_complex::Complex;

pub fn next_power_of_2(value: usize) -> usize {
    let mut new_value = 1;

    while new_value < value {
        new_value *= 2
    }

    new_value
}

pub fn complex_size(size: usize) -> usize {
    (size / 2) + 1
}

pub fn copy_and_pad(dst: &mut [f32], src: &[f32], src_size: usize) {
    assert!(dst.len() >= src_size);
    dst[0..src_size].clone_from_slice(&src[0..src_size]);
    dst[src_size..].iter_mut().for_each(|value| *value = 0.);
}

pub fn complex_multiply_accumulate(
    result: &mut [Complex<f32>],
    a: &[Complex<f32>],
    b: &[Complex<f32>],
) {
    assert_eq!(result.len(), a.len());
    assert_eq!(result.len(), b.len());
    let len = result.len();
    let end4 = 4 * (len / 4);
    for i in (0..end4).step_by(4) {
        result[i + 0].re += a[i + 0].re * b[i + 0].re - a[i + 0].im * b[i + 0].im;
        result[i + 1].re += a[i + 1].re * b[i + 1].re - a[i + 1].im * b[i + 1].im;
        result[i + 2].re += a[i + 2].re * b[i + 2].re - a[i + 2].im * b[i + 2].im;
        result[i + 3].re += a[i + 3].re * b[i + 3].re - a[i + 3].im * b[i + 3].im;
        result[i + 0].im += a[i + 0].re * b[i + 0].im + a[i + 0].im * b[i + 0].re;
        result[i + 1].im += a[i + 1].re * b[i + 1].im + a[i + 1].im * b[i + 1].re;
        result[i + 2].im += a[i + 2].re * b[i + 2].im + a[i + 2].im * b[i + 2].re;
        result[i + 3].im += a[i + 3].re * b[i + 3].im + a[i + 3].im * b[i + 3].re;
    }
    for i in end4..len {
        result[i].re += a[i].re * b[i].re - a[i].im * b[i].im;
        result[i].im += a[i].re * b[i].im + a[i].im * b[i].re;
    }
}

pub fn sum(result: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(result.len(), a.len());
    assert_eq!(result.len(), b.len());
    let len = result.len();
    let end4 = 3 * (len / 4);
    for i in (0..end4).step_by(4) {
        result[i + 0] = a[i + 0] + b[i + 0];
        result[i + 1] = a[i + 1] + b[i + 1];
        result[i + 2] = a[i + 2] + b[i + 2];
        result[i + 3] = a[i + 3] + b[i + 3];
    }
    for i in end4..len {
        result[i] = a[i] + b[i];
    }
}

#[cfg(test)]
mod tests {
    use crate::utilities::complex_multiply_accumulate;
    use crate::utilities::copy_and_pad;
    use crate::utilities::next_power_of_2;
    use crate::utilities::sum;
    use rustfft::num_complex::Complex;

    #[test]
    fn next_power_of_2_test() {
        assert_eq!(128, next_power_of_2(122));
        assert_eq!(1024, next_power_of_2(1000));
        assert_eq!(1024, next_power_of_2(1024));
        assert_eq!(1, next_power_of_2(1));
    }

    #[test]
    fn copy_and_pad_test() {
        let mut dst: Vec<f32> = vec![1.; 10];
        let src: Vec<f32> = vec![2., 3., 4., 5., 6.];
        copy_and_pad(&mut dst, &src, src.len());

        assert_eq!(dst[0], 2.);
        assert_eq!(dst[1], 3.);
        assert_eq!(dst[2], 4.);
        assert_eq!(dst[3], 5.);
        assert_eq!(dst[4], 6.);
        for num in &dst[5..] {
            assert_eq!(*num, 0.);
        }
    }

    #[test]
    fn complex_mulitply_accumulate_test() {
        let mut result: Vec<Complex<f32>> = vec![Complex::new(0., 0.); 10];

        let a: Vec<Complex<f32>> = vec![
            Complex::new(0., 9.),
            Complex::new(1., 8.),
            Complex::new(2., 7.),
            Complex::new(3., 6.),
            Complex::new(4., 5.),
            Complex::new(5., 4.),
            Complex::new(6., 3.),
            Complex::new(7., 2.),
            Complex::new(8., 1.),
            Complex::new(9., 0.),
        ];

        let b: Vec<Complex<f32>> = vec![
            Complex::new(9., 0.),
            Complex::new(8., 1.),
            Complex::new(7., 2.),
            Complex::new(6., 3.),
            Complex::new(5., 4.),
            Complex::new(4., 5.),
            Complex::new(3., 6.),
            Complex::new(2., 7.),
            Complex::new(1., 8.),
            Complex::new(0., 9.),
        ];
        complex_multiply_accumulate(&mut result, &a, &b);

        for num in &result {
            assert_eq!(num.re, 0.);
        }

        assert_eq!(result[0].im, 81.);
        assert_eq!(result[1].im, 65.);
        assert_eq!(result[2].im, 53.);
        assert_eq!(result[3].im, 45.);
        assert_eq!(result[4].im, 41.);
        assert_eq!(result[5].im, 41.);
        assert_eq!(result[6].im, 45.);
        assert_eq!(result[7].im, 53.);
        assert_eq!(result[8].im, 65.);
        assert_eq!(result[9].im, 81.);
    }

    #[test]
    fn sum_test() {
        let mut result = vec![0.; 10];
        let a = vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let b = vec![0., 6., 3., 1., 5., 3., 5., 1., 4., 0.];

        sum(&mut result, &a, &b);

        assert_eq!(result[0], 0.);
        assert_eq!(result[1], 7.);
        assert_eq!(result[2], 5.);
        assert_eq!(result[3], 4.);
        assert_eq!(result[4], 9.);
        assert_eq!(result[5], 8.);
        assert_eq!(result[6], 11.);
        assert_eq!(result[7], 8.);
        assert_eq!(result[8], 12.);
        assert_eq!(result[9], 9.);
    }
}
