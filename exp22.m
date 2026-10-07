%% =====================================================================
%  Experiment 22: Acquisition and DSP of ECG, EEG and EMG Signals
%  ---------------------------------------------------------------------
%  - Loads (or synthesises) ECG, EEG, EMG signals
%  - Removes noise using Butterworth band-pass filters (zero-phase)
%  - Plots raw vs filtered signals in time and frequency domain
%  - Extracts features: ECG R-peaks/HR, EEG band power, EMG RMS, SNR
%
%  Requires: Signal Processing Toolbox (butter, filtfilt, findpeaks,
%            pwelch, iirnotch)
%  ---------------------------------------------------------------------
%  If you have real data (kit / MIT-BIH / Physionet), set useRealData =
%  true and fill in the file names below. Each file must contain a single
%  vector variable named as given (ecg_raw, eeg_raw, emg_raw).
%% =====================================================================
clc; clear; close all;

%% ---------------- INPUT PARAMETERS (from Input Parameter Table) -------
fs_ecg = 360;            % Hz
fs_eeg = 256;            % Hz
fs_emg = 1000;           % Hz
band_ecg = [0.5  40];    % Hz
band_eeg = [1    40];    % Hz
band_emg = [20  450];    % Hz
T        = 10;           % acquisition duration (s)
mains    = 50;           % power-line frequency (Hz) - India = 50 Hz
filtOrder = 4;           % Butterworth order (per band edge)

useRealData = false;     % true -> load your own datasets
ecgFile = 'ecg_data.mat';  % contains variable  ecg_raw
eegFile = 'eeg_data.mat';  % contains variable  eeg_raw
emgFile = 'emg_data.mat';  % contains variable  emg_raw

rng(1);                  % repeatable synthetic noise

%% ---------------- 1. ACQUIRE / LOAD SIGNALS ---------------------------
if useRealData
    S = load(ecgFile); ecg_raw = S.ecg_raw(:)';
    S = load(eegFile); eeg_raw = S.eeg_raw(:)';
    S = load(emgFile); emg_raw = S.emg_raw(:)';
    % keep only first T seconds
    ecg_raw = ecg_raw(1:T*fs_ecg);
    eeg_raw = eeg_raw(1:T*fs_eeg);
    emg_raw = emg_raw(1:T*fs_emg);
else
    [ecg_raw, ecg_clean] = synth_ecg(fs_ecg, T, mains);
    [eeg_raw, eeg_clean] = synth_eeg(fs_eeg, T, mains);
    [emg_raw, emg_clean] = synth_emg(fs_emg, T, mains);
end

t_ecg = (0:numel(ecg_raw)-1)/fs_ecg;
t_eeg = (0:numel(eeg_raw)-1)/fs_eeg;
t_emg = (0:numel(emg_raw)-1)/fs_emg;

%% ---------------- 2. DIGITAL FILTERING --------------------------------
ecg_f = bp_filter(ecg_raw, band_ecg, fs_ecg, filtOrder);
eeg_f = bp_filter(eeg_raw, band_eeg, fs_eeg, filtOrder);
emg_f = bp_filter(emg_raw, band_emg, fs_emg, filtOrder);

% 20-450 Hz band does not reject 50 Hz hum, so add a notch for EMG
[bn, an] = iirnotch(mains/(fs_emg/2), (mains/(fs_emg/2))/35);
emg_f = filtfilt(bn, an, emg_f);

%% ---------------- 3. SPECTRA (FFT) ------------------------------------
[fE1, XE1] = fft_spectrum(ecg_raw, fs_ecg);  [~, XE2] = fft_spectrum(ecg_f, fs_ecg);
[fG1, XG1] = fft_spectrum(eeg_raw, fs_eeg);  [~, XG2] = fft_spectrum(eeg_f, fs_eeg);
[fM1, XM1] = fft_spectrum(emg_raw, fs_emg);  [~, XM2] = fft_spectrum(emg_f, fs_emg);

%% ---------------- 4. FEATURE EXTRACTION -------------------------------
% ---- ECG: R-peaks, RR interval, heart rate ----
[pk, loc] = findpeaks(ecg_f, 'MinPeakHeight', 0.5*max(ecg_f), ...
                      'MinPeakDistance', round(0.3*fs_ecg));
rPeakTimes = loc/fs_ecg;
RR     = diff(rPeakTimes);          % s
HR_all = 60 ./ RR;                  % beats/min  (HR = 60/RR)
HR_avg = 60 / mean(RR);

% ---- EEG: band powers (Welch PSD) ----
[Pxx, fw] = pwelch(eeg_f, hamming(2*fs_eeg), fs_eeg, 2*fs_eeg, fs_eeg);
bands = struct('name', {'Delta','Theta','Alpha','Beta'}, ...
               'range', {[1 4],[4 8],[8 13],[13 30]});
bandPower = zeros(1,4);
for k = 1:4
    idx = fw >= bands(k).range(1) & fw < bands(k).range(2);
    bandPower(k) = trapz(fw(idx), Pxx(idx));
end
relPower = 100*bandPower/sum(bandPower);
[~, domIdx] = max(bandPower);

% ---- EMG: RMS amplitude (whole record + 100 ms sliding window) ----
RMS_raw = sqrt(mean(emg_raw.^2));
RMS_f   = sqrt(mean(emg_f.^2));     % X_RMS = sqrt( (1/N) * sum x^2 )
win     = round(0.1*fs_emg);
emg_env = sqrt(movmean(emg_f.^2, win));

% ---- SNR  =  10 log10(Psignal/Pnoise) ----
% Noise estimate = what the filter removed (raw - filtered)
SNR_ecg = snr_db(ecg_f, ecg_raw - ecg_f);
SNR_eeg = snr_db(eeg_f, eeg_raw - eeg_f);
SNR_emg = snr_db(emg_f, emg_raw - emg_f);

%% ---------------- 5. PLOTS --------------------------------------------
% ===== ECG =====
figure('Name','ECG','Color','w','Position',[50 50 1100 700]);
subplot(2,2,1); plot(t_ecg, ecg_raw,'r'); grid on;
title('ECG - Raw waveform'); xlabel('Time (s)'); ylabel('Amplitude (mV)');
subplot(2,2,2); plot(t_ecg, ecg_f,'b'); hold on; plot(rPeakTimes, pk,'kv','MarkerFaceColor','g');
grid on; title(sprintf('ECG - Filtered (%.1f-%g Hz) with R-peaks', band_ecg)); 
xlabel('Time (s)'); ylabel('Amplitude (mV)'); legend('Filtered','R-peaks');
subplot(2,2,3); plot(fE1, XE1,'r'); hold on; plot(fE1, XE2,'b'); grid on; xlim([0 100]);
title('ECG - Magnitude spectrum'); xlabel('Frequency (Hz)'); ylabel('|X(f)|'); legend('Raw','Filtered');
subplot(2,2,4); stem(rPeakTimes(2:end), HR_all,'filled'); grid on; ylim([0 max(HR_all)*1.3]);
title(sprintf('Instantaneous heart rate (mean = %.1f bpm)', HR_avg));
xlabel('Time (s)'); ylabel('HR (beats/min)');

% ===== EEG =====
figure('Name','EEG','Color','w','Position',[80 80 1100 700]);
subplot(2,2,1); plot(t_eeg, eeg_raw,'r'); grid on;
title('EEG - Raw waveform'); xlabel('Time (s)'); ylabel('Amplitude (\muV)');
subplot(2,2,2); plot(t_eeg, eeg_f,'b'); grid on;
title(sprintf('EEG - Filtered (%g-%g Hz)', band_eeg)); xlabel('Time (s)'); ylabel('Amplitude (\muV)');
subplot(2,2,3); plot(fG1, XG1,'r'); hold on; plot(fG1, XG2,'b'); grid on; xlim([0 80]);
title('EEG - Magnitude spectrum'); xlabel('Frequency (Hz)'); ylabel('|X(f)|'); legend('Raw','Filtered');
subplot(2,2,4); bar(relPower); set(gca,'XTickLabel',{'Delta 1-4','Theta 4-8','Alpha 8-13','Beta 13-30'});
grid on; title('EEG - Relative band power'); ylabel('Power (%)');

% ===== EMG =====
figure('Name','EMG','Color','w','Position',[110 110 1100 700]);
subplot(2,2,1); plot(t_emg, emg_raw,'r'); grid on;
title('EMG - Raw waveform'); xlabel('Time (s)'); ylabel('Amplitude (mV)');
subplot(2,2,2); plot(t_emg, emg_f,'b'); hold on; plot(t_emg, emg_env,'k','LineWidth',1.5);
grid on; title(sprintf('EMG - Filtered (%g-%g Hz) + RMS envelope', band_emg));
xlabel('Time (s)'); ylabel('Amplitude (mV)'); legend('Filtered','RMS envelope (100 ms)');
subplot(2,2,3); plot(fM1, XM1,'r'); hold on; plot(fM1, XM2,'b'); grid on; xlim([0 500]);
title('EMG - Magnitude spectrum'); xlabel('Frequency (Hz)'); ylabel('|X(f)|'); legend('Raw','Filtered');
subplot(2,2,4); bar([RMS_raw RMS_f]); set(gca,'XTickLabel',{'Raw RMS','Filtered RMS'});
grid on; title('EMG - RMS amplitude'); ylabel('mV');

% ===== Filter frequency responses =====
figure('Name','Filter responses','Color','w','Position',[140 140 1100 400]);
[sE,gE] = bp_sos(band_ecg, fs_ecg, filtOrder);
[sG,gG] = bp_sos(band_eeg, fs_eeg, filtOrder);
[sM,gM] = bp_sos(band_emg, fs_emg, filtOrder);
[hE,wE] = freqz(sE,4096,fs_ecg); [hG,wG] = freqz(sG,4096,fs_eeg); [hM,wM] = freqz(sM,4096,fs_emg);
plot(wE,20*log10(abs(hE)+eps),'r', wG,20*log10(abs(hG)+eps),'g', wM,20*log10(abs(hM)+eps),'b','LineWidth',1.3);
grid on; xlim([0 500]); ylim([-80 5]); xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
title('Band-pass filter magnitude responses (single pass)'); legend('ECG 0.5-40','EEG 1-40','EMG 20-450');

%% ---------------- 6. OUTPUT PARAMETER TABLE ---------------------------
fprintf('\n=========== OUTPUT PARAMETER TABLE ===========\n');
fprintf('%-28s %-12s %-12s %-12s\n','Parameter','ECG','EEG','EMG');
fprintf('%-28s %-12d %-12d %-12d\n','Sampling freq (Hz)',fs_ecg,fs_eeg,fs_emg);
fprintf('%-28s %-12s %-12s %-12s\n','Filter band (Hz)', ...
        sprintf('%g-%g',band_ecg), sprintf('%g-%g',band_eeg), sprintf('%g-%g',band_emg));
fprintf('%-28s %-12d %-12d %-12d\n','Samples (10 s)',numel(ecg_raw),numel(eeg_raw),numel(emg_raw));
fprintf('%-28s %-12.2f %-12.2f %-12.2f\n','SNR after filtering (dB)',SNR_ecg,SNR_eeg,SNR_emg);

fprintf('\n--- ECG features ---\n');
fprintf('Number of R-peaks      : %d\n', numel(loc));
fprintf('Mean RR interval       : %.3f s\n', mean(RR));
fprintf('Mean heart rate        : %.1f beats/min\n', HR_avg);
fprintf('HR range (min / max)   : %.1f / %.1f beats/min\n', min(HR_all), max(HR_all));

fprintf('\n--- EEG band power ---\n');
for k = 1:4
    fprintf('%-6s (%2d-%2d Hz) : %.3f uV^2  (%.1f %%)\n', bands(k).name, ...
            bands(k).range(1), bands(k).range(2), bandPower(k), relPower(k));
end
fprintf('Dominant rhythm        : %s\n', bands(domIdx).name);

fprintf('\n--- EMG features ---\n');
fprintf('RMS (raw)              : %.4f mV\n', RMS_raw);
fprintf('RMS (filtered)         : %.4f mV\n', RMS_f);
fprintf('Peak RMS envelope      : %.4f mV\n', max(emg_env));

if ~useRealData
    fprintf('\n--- Check against known clean synthetic signals ---\n');
    fprintf('ECG SNR raw  : %.2f dB | filtered : %.2f dB\n', snr_db(ecg_clean, ecg_raw-ecg_clean), snr_db(ecg_clean, ecg_f-ecg_clean));
    fprintf('EEG SNR raw  : %.2f dB | filtered : %.2f dB\n', snr_db(eeg_clean, eeg_raw-eeg_clean), snr_db(eeg_clean, eeg_f-eeg_clean));
    fprintf('EMG SNR raw  : %.2f dB | filtered : %.2f dB\n', snr_db(emg_clean, emg_raw-emg_clean), snr_db(emg_clean, emg_f-emg_clean));
end

%% =====================================================================
%                          LOCAL FUNCTIONS
%% =====================================================================
function [sos, g] = bp_sos(band, fs, order)
    [z,p,k] = butter(order, band/(fs/2), 'bandpass');
    [sos, g] = zp2sos(z,p,k);          % second-order sections: stable for 0.5 Hz edge
end

function y = bp_filter(x, band, fs, order)
    [sos, g] = bp_sos(band, fs, order);
    y = filtfilt(sos, g, x);           % zero-phase (no waveform distortion)
end

function [f, X] = fft_spectrum(x, fs)
    N = numel(x);
    Xf = fft(x - mean(x));             % X[k] = sum x[n] e^{-j2pi kn/N}
    X = abs(Xf(1:floor(N/2)+1))/N;
    X(2:end-1) = 2*X(2:end-1);         % single-sided amplitude
    f = (0:floor(N/2))*fs/N;
end

function s = snr_db(sig, noise)
    s = 10*log10(sum(sig.^2)/sum(noise.^2));   % 10 log10(Ps/Pn)
end

% ----------------------- synthetic signal generators -------------------
function [x, clean] = synth_ecg(fs, T, mains)
    t = (0:1/fs:T-1/fs);
    clean = zeros(size(t));
    beatT = 0.4; HRv = 0;
    while beatT < T
        % P, Q, R, S, T waves as Gaussians: [amp(mV), centre(s), width(s)]
        w = [ 0.15 -0.20 0.025;
             -0.15 -0.040 0.010;
              1.20  0.000 0.012;
             -0.25  0.040 0.012;
              0.30  0.250 0.050];
        for i = 1:5
            clean = clean + w(i,1)*exp(-((t-beatT-w(i,2)).^2)/(2*w(i,3)^2));
        end
        beatT = beatT + 0.83 + 0.03*randn;     % RR ~ 0.83 s (about 72 bpm)
    end
    base  = 0.5*sin(2*pi*0.25*t);                % baseline wander
    hum   = 0.15*sin(2*pi*mains*t);              % power-line interference
    noise = 0.03*randn(size(t));                 % broadband noise
    x = clean + base + hum + noise;
end

function [x, clean] = synth_eeg(fs, T, mains)
    t = (0:1/fs:T-1/fs);
    clean = 12*sin(2*pi*2*t+1) + 8*sin(2*pi*6*t+2) ...   % delta, theta
          + 25*sin(2*pi*10*t) + 6*sin(2*pi*20*t+0.5);    % alpha, beta
    clean = clean + 3*randn(size(t));
    drift = 40*sin(2*pi*0.1*t);                          % electrode drift
    hum   = 10*sin(2*pi*mains*t);                        % mains
    x = clean + drift + hum;
end

function [x, clean] = synth_emg(fs, T, mains)
    t = (0:1/fs:T-1/fs);
    n = randn(size(t));
    [sos,g] = bp_sos([20 450], fs, 4);
    n = filtfilt(sos, g, n);                             % EMG-like spectrum
    env = zeros(size(t));                                % three contractions
    for c = [1.5 4.5 7.5]
        env = env + exp(-((t-c).^2)/(2*0.5^2));
    end
    clean = 0.5*env.*n;
    motion = 0.3*sin(2*pi*3*t);                          % motion artefact
    hum    = 0.1*sin(2*pi*mains*t);
    x = clean + motion + hum + 0.01*randn(size(t));
end