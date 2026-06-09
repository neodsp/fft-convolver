use audio_blocks::Planar;
use audio_host::{AudioBackend, AudioBlock, AudioBlockOpsMut, Config, Error};
use fft_convolver::FFTConvolver;

const SAMPLE_RATE: u32 = 48_000;
const BLOCK_SIZE: usize = 1024;
const NUM_CHANNELS: u16 = 2;
const CUTOFF_HZ: f32 = 500.0;
const NUM_TAPS: usize = 1023;

fn highpass_fir(cutoff_hz: f32, sample_rate: f32, num_taps: usize) -> Vec<f32> {
    let fc = cutoff_hz / sample_rate;
    let m = num_taps - 1;
    let center = m as f32 / 2.0;
    let pi = std::f32::consts::PI;

    let mut h: Vec<f32> = (0..num_taps)
        .map(|n| {
            let x = n as f32 - center;
            let window = 0.5 * (1.0 - (2.0 * pi * n as f32 / m as f32).cos());
            let sinc = if x.abs() < 1e-6 {
                2.0 * fc
            } else {
                (2.0 * pi * fc * x).sin() / (pi * x)
            };
            sinc * window
        })
        .collect();

    // Spectral inversion: highpass = delta - lowpass
    h.iter_mut().for_each(|s| *s = -*s);
    h[m / 2] += 1.0;
    h
}

fn main() -> Result<(), Error> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: highpass_playback <audio-file>");
        std::process::exit(1);
    });

    let audio = audio_file::read::<f32>(
        &path,
        audio_file::ReadConfig {
            sample_rate: Some(SAMPLE_RATE),
            num_channels: Some(NUM_CHANNELS as usize),
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("Failed to read audio file: {e}");
        std::process::exit(1);
    });

    let num_frames_total = audio.samples_interleaved.len() / NUM_CHANNELS as usize;
    let duration_secs = num_frames_total as f64 / SAMPLE_RATE as f64;
    let tail_secs = BLOCK_SIZE as f64 / SAMPLE_RATE as f64;

    println!(
        "Playing '{}' ({:.1}s) with {CUTOFF_HZ} Hz highpass filter...",
        path, duration_secs
    );

    let ir = highpass_fir(CUTOFF_HZ, SAMPLE_RATE as f32, NUM_TAPS);

    let mut convolvers: Vec<FFTConvolver<f32>> = (0..NUM_CHANNELS)
        .map(|_| {
            let mut c = FFTConvolver::default();
            c.init(BLOCK_SIZE, &ir).expect("convolver init failed");
            c
        })
        .collect();

    let mut input_block = Planar::<f32>::new(NUM_CHANNELS, BLOCK_SIZE);
    let mut output_block = Planar::<f32>::new(NUM_CHANNELS, BLOCK_SIZE);
    let audio_data = audio.samples_interleaved;
    let mut pos = 0usize;

    let mut host = audio_host::AudioHost::new()?;

    host.start(
        Config {
            num_input_channels: 0,
            num_output_channels: NUM_CHANNELS,
            sample_rate: SAMPLE_RATE,
            num_frames: BLOCK_SIZE,
        },
        move |_input, mut output| {
            let num_frames = output.num_frames() as usize;
            let num_ch = output.num_channels() as usize;

            for ch in 0..num_ch {
                let in_ch = input_block.channel_mut(ch as u16);
                for f in 0..num_frames {
                    let idx = (pos + f) * num_ch + ch;
                    in_ch[f] = if idx < audio_data.len() {
                        audio_data[idx]
                    } else {
                        0.0
                    };
                }
                convolvers[ch]
                    .process(
                        input_block.channel(ch as u16),
                        output_block.channel_mut(ch as u16),
                    )
                    .expect("process failed");
            }

            output.copy_from_block_exact(&output_block);

            pos += num_frames;
        },
    )?;

    std::thread::sleep(std::time::Duration::from_secs_f64(
        duration_secs + tail_secs,
    ));

    host.stop()?;
    println!("Done.");

    Ok(())
}
