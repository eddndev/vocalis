use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;


// Helper function to convert Hertz to Mel scale
fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

// Helper function to convert Mel scale to Hertz
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

pub struct DspProcessor {
    sample_rate: u32,
    pub n_mfcc: usize,
    pub frame_length: usize, // Window length in samples
    pub hop_length: usize,   // Hop length in samples
    n_fft: usize,        // FFT size
    n_mels: usize,       // Number of Mel bands
    f_min: f32,          // Minimum frequency for Mel filterbank
    f_max: f32,          // Maximum frequency for Mel filterbank
    pub fft_forward: std::sync::Arc<dyn rustfft::Fft<f32>>, // Planner for forward FFT
}

impl DspProcessor {
    pub fn new(sample_rate: u32, n_mfcc: usize) -> Self {
        let frame_length_ms = 25.0; // 25ms
        let hop_length_ms = 10.0;   // 10ms

        let frame_length = (sample_rate as f32 * frame_length_ms / 1000.0).round() as usize;
        let hop_length = (sample_rate as f32 * hop_length_ms / 1000.0).round() as usize;
        
        // n_fft is typically the smallest power of 2 greater than or equal to frame_length
        let n_fft = frame_length.next_power_of_two();
        
        let n_mels = 40; // Common value
        let f_min = 20.0;
        let f_max = sample_rate as f32 / 2.0;

        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(n_fft);

        DspProcessor {
            sample_rate,
            n_mfcc,
            frame_length,
            hop_length,
            n_fft,
            n_mels,
            f_min,
            f_max,
            fft_forward,
        }
    }

    pub fn pre_emphasis(&self, samples: &mut [f32], pre_emphasis_coeff: f32) {
        if samples.is_empty() {
            return;
        }
        for i in (1..samples.len()).rev() {
            samples[i] -= pre_emphasis_coeff * samples[i - 1];
        }
    }

    pub fn hamming_window(&self, frame: &mut [f32]) {
        let n = frame.len();
        if n == 0 { return; }
        for i in 0..n {
            // Changed to HANN window to match Librosa default
            // Formula: 0.5 * (1 - cos(2*pi*n/(N-1)))
            let window_val = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos());
            frame[i] *= window_val;
        }
    }

    pub fn power_spectrum(&self, frame: &[f32]) -> Vec<f32> {
        let mut buffer: Vec<Complex<f32>> = frame.iter().map(|&x| Complex::new(x, 0.0)).collect();
        buffer.resize(self.n_fft, Complex::new(0.0, 0.0));

        self.fft_forward.process(&mut buffer);

        // Compute power spectrum (magnitude squared)
        let num_bins = self.n_fft / 2 + 1;
        buffer[0..num_bins].iter().map(|c| c.norm_sqr()).collect()
    }

    pub fn mel_filter_bank(&self, power_spectrum: &[f32]) -> Vec<f32> {
        let _max_freq = self.sample_rate as f32 / 2.0;
        let mut mel_points = Vec::with_capacity(self.n_mels + 2);
        
        // Generate Mel points
        let mel_min = hz_to_mel(self.f_min);
        let mel_max = hz_to_mel(self.f_max);
        for i in 0..self.n_mels + 2 {
            let mel_i = mel_min + (mel_max - mel_min) / (self.n_mels + 1) as f32 * i as f32;
            mel_points.push(mel_to_hz(mel_i));
        }

        // Convert Mel points to FFT bin numbers
        // BUG FIX: Correct mapping implies indices within 0..N_FFT/2
        // Formula: bin = freq * (N_FFT + 1) / SampleRate
        // (Using N_FFT+1 to closely match Librosa's linspace behavior logic)
        let fft_bins: Vec<usize> = mel_points.iter()
            .map(|&p| p * (self.n_fft as f32 + 1.0) / self.sample_rate as f32)
            .map(|p| p.floor() as usize)
            .collect();

        let mut mel_energies = vec![0.0; self.n_mels];
        let num_spectrum_bins = power_spectrum.len();

        for i in 0..self.n_mels {
            let m_low = fft_bins[i];
            let m_center = fft_bins[i + 1];
            let m_high = fft_bins[i + 2];

            let f_low = mel_points[i];
            let f_high = mel_points[i+2];
            // Slaney normalization factor: 2 / (f_high - f_low)
            let norm_factor = 2.0 / (f_high - f_low);

            if m_center == m_low || m_high == m_center {
                continue; // Avoid division by zero
            }

            // Up-slope
            for j in m_low..m_center {
                if j < num_spectrum_bins {
                    let weight = (j as f32 - m_low as f32) / (m_center as f32 - m_low as f32);
                    mel_energies[i] += power_spectrum[j] * weight * norm_factor;
                }
            }

            // Down-slope
            for j in m_center..m_high {
                if j < num_spectrum_bins {
                    let weight = (m_high as f32 - j as f32) / (m_high as f32 - m_center as f32);
                    mel_energies[i] += power_spectrum[j] * weight * norm_factor;
                }
            }
        }
        mel_energies
    }

    pub fn log_energies(&self, mel_energies: &[f32]) -> Vec<f32> {
        mel_energies.iter().map(|&e| 10.0 * (e + f32::EPSILON).log10()).collect()
    }

    pub fn dct(&self, log_mel_energies: &[f32]) -> Vec<f32> {
        let mut mfccs = vec![0.0; self.n_mfcc];
        let num_filters = log_mel_energies.len();
        let sqrt_2_n = (2.0 / num_filters as f32).sqrt();
        let sqrt_1_n = (1.0 / num_filters as f32).sqrt();

        for i in 0..self.n_mfcc {
            let mut sum = 0.0;
            for j in 0..num_filters {
                sum += log_mel_energies[j] * ((PI * i as f32 / num_filters as f32) * (j as f32 + 0.5)).cos();
            }
            
            if i == 0 {
                mfccs[i] = sum * sqrt_1_n;
            } else {
                mfccs[i] = sum * sqrt_2_n;
            }
        }
        mfccs
    }

    #[allow(dead_code)]
    pub fn cepstral_mean_normalization(&self, mfccs: &mut [f32]) {
        if mfccs.is_empty() {
            return;
        }
        let mean = mfccs.iter().sum::<f32>() / mfccs.len() as f32;
        for mfcc in mfccs.iter_mut() {
            *mfcc -= mean;
        }
    }

    pub fn extract_features(&self, audio_frame: &[f32], pre_emphasis_coeff: f32) -> Vec<f32> {
        let mut frame = audio_frame.to_vec();

        // 1. Pre-emphasis
        self.pre_emphasis(&mut frame, pre_emphasis_coeff);

        // 2. Hamming Window
        self.hamming_window(&mut frame);

        // 3. Power Spectrum (FFT)
        let power_spectrum = self.power_spectrum(&frame);

        // 4. Mel Filter Bank
        let mel_energies = self.mel_filter_bank(&power_spectrum);

        // 5. Log Energies
        let log_mel_energies = self.log_energies(&mel_energies);

        // 6. DCT (MFCCs)
        let mfccs = self.dct(&log_mel_energies);

        // NOTE: CMN (Cepstral Mean Normalization) removed.
        // We rely on Global Standardization (Scaler) in the inference stage.
        // self.cepstral_mean_normalization(&mut mfccs);

        mfccs
    }

    pub fn compute_pitch(&self, audio_data: &[f32]) -> f32 {
        let f_min = 50.0;
        let f_max = 400.0;
        let win_length = 2048;
        let hop_length = 512;
        
        let mut pitches = Vec::new();

        if audio_data.len() < win_length {
            return 0.0;
        }

        let mut i = 0;
        while i + win_length <= audio_data.len() {
            let window = &audio_data[i..i+win_length];
            
            // Autocorrelation
            let mut r = vec![0.0; win_length];
            for lag in 0..win_length {
                let mut sum = 0.0;
                for j in 0..win_length - lag {
                    sum += window[j] * window[j + lag];
                }
                r[lag] = sum;
            }

            // Find peak in range
            let min_lag = (self.sample_rate as f32 / f_max) as usize;
            let max_lag = (self.sample_rate as f32 / f_min) as usize;
            
            let mut max_corr = -1.0;
            let mut best_lag = 0;

            for lag in min_lag..=max_lag {
                if lag < r.len() && r[lag] > max_corr {
                    max_corr = r[lag];
                    best_lag = lag;
                }
            }

            if best_lag > 0 {
                pitches.push(self.sample_rate as f32 / best_lag as f32);
            }

            i += hop_length;
        }

        if pitches.is_empty() {
            return 0.0;
        }

        pitches.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        pitches[pitches.len() / 2]
    }

    /// Extract 39-dimensional feature vector for syllable classification.
    /// 
    /// Segments the audio into 3 temporal regions:
    /// - Onset (35%): Captures consonant characteristics
    /// - Transition (15%): Captures coarticulation
    /// - Nucleus (50%): Captures the vowel
    /// 
    /// Each region gets 13 MFCCs (averaged over time), resulting in 39 total features.
    pub fn extract_syllable_features(&self, audio_data: &[f32], pre_emphasis_coeff: f32) -> Vec<f32> {
        let total_samples = audio_data.len();
        
        // Ensure minimum length
        if total_samples < self.frame_length * 3 {
            // Pad audio if too short
            let mut padded = audio_data.to_vec();
            padded.resize(self.frame_length * 3, 0.0);
            return self.extract_syllable_features(&padded, pre_emphasis_coeff);
        }
        
        // Calculate split points (35% / 15% / 50%)
        let onset_end = (total_samples as f32 * 0.35) as usize;
        let trans_end = (total_samples as f32 * 0.50) as usize;
        
        // Split audio into 3 regions
        let onset_audio = &audio_data[..onset_end];
        let trans_audio = &audio_data[onset_end..trans_end];
        let nucleus_audio = &audio_data[trans_end..];
        
        // Extract MFCCs for each region (using existing bag-of-frames approach)
        let mfcc_onset = self.extract_region_mfccs(onset_audio, pre_emphasis_coeff);
        let mfcc_trans = self.extract_region_mfccs(trans_audio, pre_emphasis_coeff);
        let mfcc_nucleus = self.extract_region_mfccs(nucleus_audio, pre_emphasis_coeff);
        
        // Concatenate into 39-dim vector
        let mut features = Vec::with_capacity(39);
        features.extend_from_slice(&mfcc_onset);
        features.extend_from_slice(&mfcc_trans);
        features.extend_from_slice(&mfcc_nucleus);
        
        features
    }
    
    /// Helper: Extract averaged MFCCs from a region of audio
    fn extract_region_mfccs(&self, audio_region: &[f32], pre_emphasis_coeff: f32) -> Vec<f32> {
        let mut all_mfccs: Vec<Vec<f32>> = Vec::new();
        
        // Ensure minimum size
        let region = if audio_region.len() < self.frame_length {
            let mut padded = audio_region.to_vec();
            padded.resize(self.frame_length, 0.0);
            padded
        } else {
            audio_region.to_vec()
        };
        
        // Frame and extract MFCCs
        let mut current_sample = 0;
        while current_sample + self.frame_length <= region.len() {
            let frame = &region[current_sample..current_sample + self.frame_length];
            let mfccs = self.extract_features(frame, pre_emphasis_coeff);
            all_mfccs.push(mfccs);
            current_sample += self.hop_length;
        }
        
        // If no frames, extract from padded frame
        if all_mfccs.is_empty() {
            let mut padded = region.clone();
            padded.resize(self.frame_length, 0.0);
            all_mfccs.push(self.extract_features(&padded, pre_emphasis_coeff));
        }
        
        // Average across frames (bag-of-frames)
        let mut avg_mfccs = vec![0.0f32; self.n_mfcc];
        for mfcc_vec in &all_mfccs {
            for (i, &val) in mfcc_vec.iter().enumerate() {
                avg_mfccs[i] += val;
            }
        }
        for val in avg_mfccs.iter_mut() {
            *val /= all_mfccs.len() as f32;
        }
        
        avg_mfccs
    }
}

