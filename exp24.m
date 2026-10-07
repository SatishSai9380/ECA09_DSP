%% =====================================================================
%  OFDM Channel Estimation using LS and MMSE Techniques
%  ---------------------------------------------------------------------
%  Subcarriers : 64            Pilots (subcarrier index) : [8 22 44 58]
%  Modulation  : QPSK (4-QAM)  Channel : 4-tap Rayleigh fading
%  SNR         : 20 dB         (BER/MSE also swept over 0-30 dB)
%
%  Notes
%  - Written without Communications Toolbox (QPSK mapping done by hand).
%    Signal Processing Toolbox is not required either (only interp1).
%  - Each OFDM symbol sees an independent Rayleigh channel realisation
%    (constant over the symbol, so the cyclic prefix makes it a
%    per-subcarrier multiplication).
%  - Es = 1, noise variance per subcarrier = 10^(-SNR/10).
%% =====================================================================
clc; clear; close all;
rng(1);

%% ---------------- INPUT PARAMETERS ------------------------------------
P.N       = 64;                    % number of subcarriers
P.pilots  = [8 22 44 58];          % pilot subcarrier indices (1-based)
P.Lcp     = 16;                    % cyclic prefix length (> channel length)
P.Xp      = [1+1j; 1-1j; -1+1j; -1-1j]/sqrt(2);   % known unit-power QPSK pilots
P.dataIdx = setdiff(1:P.N, P.pilots);
SNR_dB    = 20;                    % operating SNR for detailed plots
NsymMain  = 3000;                  % OFDM symbols for the 20 dB run
NsymSweep = 4000;                  % OFDM symbols per SNR point in the sweep
snrSweep  = 0:5:30;                % SNR sweep (dB)

% 4-tap power-delay profile (exponential decay, total power = 1)
pdp   = exp(-(0:3)/1.5);  pdp = pdp/sum(pdp);
P.pdp = pdp(:);

% Channel frequency correlation matrix R_HH (N x N) from the PDP:
%   R(k,m) = sum_l pdp(l) * exp(-j*2*pi*(k-m)*l/N)
dk = (0:P.N-1).' - (0:P.N-1);
R  = zeros(P.N);
for l = 0:numel(pdp)-1
    R = R + pdp(l+1)*exp(-1j*2*pi*dk*l/P.N);
end
P.R = R;

%% ---------------- 1. MAIN RUN AT 20 dB --------------------------------
out = run_ofdm(SNR_dB, NsymMain, P);

%% ---------------- 2. BER / MSE SWEEP ----------------------------------
berLS = zeros(size(snrSweep)); berMMSE = berLS; berPerf = berLS;
mseLS = berLS; mseMMSE = berLS;
for n = 1:numel(snrSweep)
    o = run_ofdm(snrSweep(n), NsymSweep, P);
    berLS(n) = o.ber(1);  berMMSE(n) = o.ber(2);  berPerf(n) = o.ber(3);
    mseLS(n) = o.mse(1);  mseMMSE(n) = o.mse(2);
end
gam   = 10.^(snrSweep/10)/2;                 % Eb/N0 for QPSK (Es/N0 = SNR)
berTh = 0.5*(1 - sqrt(gam./(1+gam)));        % QPSK over flat Rayleigh, ideal CSI

%% ---------------- 3. PLOTS --------------------------------------------
k = 1:P.N;

% ---- Figure 1: channel estimates for one OFDM symbol ----
figure('Name','Channel estimates','Color','w','Position',[40 40 1100 450]);
subplot(1,2,1);
plot(k, abs(out.H1(:,1)),'k','LineWidth',1.8); hold on;
plot(k, abs(out.H1(:,2)),'r--','LineWidth',1.3);
plot(k, abs(out.H1(:,3)),'b-.','LineWidth',1.3);
plot(P.pilots, abs(out.Hpil1),'go','MarkerFaceColor','g','MarkerSize',7);
grid on; xlabel('Subcarrier index'); ylabel('|H(k)|');
title(sprintf('Channel magnitude (SNR = %d dB)', SNR_dB));
legend('Actual','LS (interpolated)','MMSE','LS at pilots','Location','best');
subplot(1,2,2);
plot(k, angle(out.H1(:,1)),'k','LineWidth',1.8); hold on;
plot(k, angle(out.H1(:,2)),'r--','LineWidth',1.3);
plot(k, angle(out.H1(:,3)),'b-.','LineWidth',1.3);
grid on; xlabel('Subcarrier index'); ylabel('Phase (rad)');
title('Channel phase'); legend('Actual','LS','MMSE','Location','best');

% ---- Figure 2: estimation error per subcarrier ----
figure('Name','Estimation error','Color','w','Position',[80 80 1100 420]);
subplot(1,2,1);
semilogy(k, out.mse_k(:,1),'r-o','MarkerSize',4); hold on;
semilogy(k, out.mse_k(:,2),'b-s','MarkerSize',4);
xline(P.pilots,':k');
grid on; xlabel('Subcarrier index'); ylabel('MSE = E|H - H_{est}|^2');
title(sprintf('Per-subcarrier estimation error (SNR = %d dB)', SNR_dB));
legend('LS','MMSE','Location','best');
subplot(1,2,2);
semilogy(snrSweep, mseLS,'r-o','LineWidth',1.4); hold on;
semilogy(snrSweep, mseMMSE,'b-s','LineWidth',1.4);
grid on; xlabel('SNR (dB)'); ylabel('Channel estimation MSE');
title('Estimation MSE vs SNR'); legend('LS','MMSE','Location','southwest');

% ---- Figure 3: equalised constellations ----
figure('Name','Constellations','Color','w','Position',[120 120 1200 400]);
tit = {'Perfect CSI','LS estimate','MMSE estimate'};
fn  = {'perfect','ls','mmse'};
for i = 1:3
    subplot(1,3,i);
    z = out.const.(fn{i});
    plot(real(z), imag(z), '.', 'MarkerSize', 3); hold on;
    ideal = [1+1j 1-1j -1+1j -1-1j]/sqrt(2);
    plot(real(ideal), imag(ideal),'r+','MarkerSize',12,'LineWidth',2);
    grid on; axis equal; axis([-2 2 -2 2]);
    xlabel('In-phase'); ylabel('Quadrature');
    berIdx = [3 1 2];                      % out.ber = [LS MMSE perfect]
    title(sprintf('%s (BER = %.2e)', tit{i}, out.ber(berIdx(i))));
end

% ---- Figure 4: BER vs SNR ----
figure('Name','BER comparison','Color','w','Position',[160 160 700 500]);
semilogy(snrSweep, berLS,'r-o','LineWidth',1.5); hold on;
semilogy(snrSweep, berMMSE,'b-s','LineWidth',1.5);
semilogy(snrSweep, berPerf,'k-^','LineWidth',1.5);
semilogy(snrSweep, berTh,'g--','LineWidth',1.2);
grid on; xlabel('SNR (dB)'); ylabel('Bit error rate');
title('BER: LS vs MMSE channel estimation'); ylim([1e-4 1]);
legend('LS','MMSE','Perfect CSI','Theory (Rayleigh, ideal CSI)','Location','southwest');

%% ---------------- 4. OUTPUT PARAMETER TABLE ---------------------------
fprintf('\n=========== OUTPUT PARAMETER TABLE (SNR = %d dB) ===========\n', SNR_dB);
fprintf('Channel at pilot subcarriers (first OFDM symbol)\n');
fprintf('%-8s %-24s %-24s %-24s\n','Pilot','Actual H','LS estimate','MMSE estimate');
for i = 1:numel(P.pilots)
    p = P.pilots(i);
    fprintf('%-8d %-24s %-24s %-24s\n', p, cstr(out.H1(p,1)), cstr(out.H1(p,2)), cstr(out.H1(p,3)));
end
fprintf('\n%-30s %-14s %-14s\n','Metric','LS','MMSE');
fprintf('%-30s %-14.4e %-14.4e\n','Channel estimation MSE', out.mse(1), out.mse(2));
fprintf('%-30s %-14.4e %-14.4e\n','Bit error rate (BER)', out.ber(1), out.ber(2));
fprintf('Perfect-CSI BER                : %.4e\n', out.ber(3));
fprintf('Bits simulated at %d dB         : %d\n', SNR_dB, 2*numel(P.dataIdx)*NsymMain);

fprintf('\n%-10s %-12s %-12s %-12s %-12s %-12s\n','SNR (dB)','BER LS','BER MMSE','BER perfect','MSE LS','MSE MMSE');
for n = 1:numel(snrSweep)
    fprintf('%-10d %-12.3e %-12.3e %-12.3e %-12.3e %-12.3e\n', snrSweep(n), ...
        berLS(n), berMMSE(n), berPerf(n), mseLS(n), mseMMSE(n));
end

%% =====================================================================
%                          LOCAL FUNCTIONS
%% =====================================================================
function s = cstr(z)
    s = sprintf('%.3f%+.3fj', real(z), imag(z));
end

function out = run_ofdm(snrdB, Nsym, P)
    N = P.N;  pil = P.pilots(:);  Np = numel(pil);
    dIdx = P.dataIdx(:);  Nd = numel(dIdx);  L = numel(P.pdp);
    sig2 = 10^(-snrdB/10);                      % noise variance (Es = 1)

    % ---- transmitter: QPSK data + pilots ----
    b1 = randi([0 1], Nd, Nsym);  b2 = randi([0 1], Nd, Nsym);
    dataSym = ((1-2*b1) + 1j*(1-2*b2))/sqrt(2);
    X = zeros(N, Nsym);
    X(dIdx,:) = dataSym;
    X(pil,:)  = repmat(P.Xp(:), 1, Nsym);
    x   = sqrt(N)*ifft(X);                      % unit-power time samples
    xcp = [x(N-P.Lcp+1:N,:); x];                % add cyclic prefix

    % ---- 4-tap Rayleigh channel + AWGN ----
    h = sqrt(P.pdp/2) .* (randn(L,Nsym) + 1j*randn(L,Nsym));
    y = zeros(N+P.Lcp, Nsym);
    for s = 1:Nsym
        y(:,s) = filter(h(:,s), 1, xcp(:,s));
    end
    y = y + sqrt(sig2/2)*(randn(size(y)) + 1j*randn(size(y)));

    % ---- receiver: remove CP, FFT ----
    Y = fft(y(P.Lcp+1:end,:))/sqrt(N);
    Hact = fft(h, N, 1);                        % true frequency response

    % ---- LS estimate at pilots:  H_LS = Y / X ----
    Hls_p = Y(pil,:) ./ P.Xp(:);
    Hls   = interp1(pil, Hls_p, (1:N).', 'linear', 'extrap');   % all subcarriers

    % ---- MMSE estimate: R_Hp * (R_pp + sigma^2 I)^-1 * H_LS,p ----
    Rpp = P.R(pil,pil);
    Rfp = P.R(:,pil);
    W   = Rfp / (Rpp + (sig2/1)*eye(Np));       % |Xp|^2 = 1
    Hmmse = W * Hls_p;

    % ---- equalisation (zero-forcing, one-tap) ----
    Yd = Y(dIdx,:);
    Xe.perfect = Yd ./ Hact(dIdx,:);
    Xe.ls      = Yd ./ Hls(dIdx,:);
    Xe.mmse    = Yd ./ Hmmse(dIdx,:);

    % ---- BER ----
    nbits = 2*Nd*Nsym;
    f = fieldnames(Xe);  ber = zeros(1,3);
    for i = 1:3
        z = Xe.(f{i});
        err = sum(sum((real(z)<0) ~= (b1==1))) + sum(sum((imag(z)<0) ~= (b2==1)));
        ber(i) = err/nbits;
    end
    out.ber = [ber(2) ber(3) ber(1)];           % [LS MMSE perfect]

    % ---- channel estimation MSE = (1/N) sum |H - Hest|^2 ----
    eL = abs(Hact - Hls).^2;   eM = abs(Hact - Hmmse).^2;
    out.mse   = [mean(eL(:)) mean(eM(:))];
    out.mse_k = [mean(eL,2) mean(eM,2)];

    % ---- items for plotting ----
    out.H1    = [Hact(:,1) Hls(:,1) Hmmse(:,1)];
    out.Hpil1 = Hls_p(:,1);
    nPlot = min(300, Nsym);
    out.const.perfect = reshape(Xe.perfect(:,1:nPlot), [], 1);
    out.const.ls      = reshape(Xe.ls(:,1:nPlot), [], 1);
    out.const.mmse    = reshape(Xe.mmse(:,1:nPlot), [], 1);
end