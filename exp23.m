%% =====================================================================
%  DSP-Based Image Enhancement, Compression and Edge Detection
%  ---------------------------------------------------------------------
%  Test image   : cameraman.tif (256x256 grayscale)
%  Spatial filter: 3x3 median filter
%  Enhancement  : contrast stretching using imadjust
%  Compression  : JPEG, quality = 50
%  Edge detector: Canny, thresholds = [0.1 0.2]
%  Requires     : Image Processing Toolbox
%% =====================================================================
clc; clear; close all;

%% ---------------- INPUT PARAMETERS ------------------------------------
imgName     = 'cameraman.tif';
medSize     = [3 3];          % median filter window
jpegQuality = 50;             % JPEG quality factor
cannyTh     = [0.1 0.2];      % Canny thresholds [low high]
MAXI        = 255;            % peak value for 8-bit image

%% ---------------- 1. LOAD IMAGE ---------------------------------------
I = imread(imgName);
if size(I,3) == 3, I = rgb2gray(I); end
[rows, cols] = size(I);
fprintf('Image: %s | Size: %d x %d | Class: %s\n', imgName, rows, cols, class(I));
fprintf('Min = %d, Max = %d, Mean = %.2f, Std = %.2f\n', ...
        min(I(:)), max(I(:)), mean(double(I(:))), std(double(I(:))));

%% ---------------- 2. SPATIAL FILTERING + ENHANCEMENT ------------------
I_med = medfilt2(I, medSize);                       % 3x3 median filter (noise smoothing)
I_enh = imadjust(I_med, stretchlim(I_med), []);     % contrast stretching

%% ---------------- 3. JPEG COMPRESSION AND RECONSTRUCTION --------------
jpgFile = fullfile(tempdir, 'cameraman_q50.jpg');
imwrite(I, jpgFile, 'jpg', 'Quality', jpegQuality);
I_rec = imread(jpgFile);                            % reconstructed image

origBytes = rows*cols*1;                            % 8 bit/pixel = 1 byte/pixel
info      = dir(jpgFile);
compBytes = info.bytes;
CR        = origBytes / compBytes;                  % compression ratio
bpp       = 8*compBytes/(rows*cols);                % bits per pixel after compression

%% ---------------- 4. EDGE DETECTION (CANNY) ---------------------------
E_orig = edge(I,     'canny', cannyTh);
E_enh  = edge(I_enh, 'canny', cannyTh);
E_rec  = edge(I_rec, 'canny', cannyTh);

%% ---------------- 5. QUALITY METRICS ----------------------------------
MSE  = @(A,B) mean((double(A(:)) - double(B(:))).^2);   % MSE = (1/N) sum (I - I^)^2
PSNR = @(A,B) 10*log10(MAXI^2 / MSE(A,B));              % PSNR = 10 log10(MAX^2/MSE)

MSE_med = MSE(I, I_med);   PSNR_med = PSNR(I, I_med);
MSE_enh = MSE(I, I_enh);   PSNR_enh = PSNR(I, I_enh);
MSE_rec = MSE(I, I_rec);   PSNR_rec = PSNR(I, I_rec);

NIE_med = MSE_med / MAXI^2;      % normalized image error = MSE/MAXI^2
NIE_enh = MSE_enh / MAXI^2;
NIE_rec = MSE_rec / MAXI^2;

SSIM_rec = ssim(I_rec, I);       % extra metric (structural similarity)

edgePix = [nnz(E_orig) nnz(E_enh) nnz(E_rec)];

%% ---------------- 6. DISPLAY RESULTS ----------------------------------
figure('Name','Enhancement, Compression, Edge Detection','Color','w', ...
       'Position',[40 40 1200 750]);
subplot(2,3,1); imshow(I);      title('Original image');
subplot(2,3,2); imshow(I_med);  title('After 3x3 median filter');
subplot(2,3,3); imshow(I_enh);  title('Enhanced (median + imadjust)');
subplot(2,3,4); imshow(I_rec);  title(sprintf('JPEG Q=%d reconstructed (CR = %.2f)', jpegQuality, CR));
subplot(2,3,5); imshow(E_orig); title('Canny edge map (original)');
subplot(2,3,6); imshow(E_rec);  title('Canny edge map (JPEG reconstructed)');

% Histograms: effect of contrast stretching
figure('Name','Histograms','Color','w','Position',[80 80 900 350]);
subplot(1,2,1); imhist(I);     title('Histogram - original');  ylim auto;
subplot(1,2,2); imhist(I_enh); title('Histogram - enhanced');  ylim auto;

% Difference (error) image for compression
figure('Name','Compression error','Color','w','Position',[120 120 900 380]);
subplot(1,2,1); imshow(abs(double(I)-double(I_rec)), []); colorbar;
title('|Original - JPEG reconstructed|');
subplot(1,2,2); imshow(E_enh); title('Canny edge map (enhanced image)');

% Rate-distortion curve for several JPEG qualities
qList = 10:10:90;
psnrQ = zeros(size(qList)); crQ = zeros(size(qList));
for n = 1:numel(qList)
    f = fullfile(tempdir, 'tmp_q.jpg');
    imwrite(I, f, 'jpg', 'Quality', qList(n));
    R = imread(f);
    d = dir(f);
    psnrQ(n) = PSNR(I, R);
    crQ(n)   = origBytes / d.bytes;
end
figure('Name','Rate-distortion','Color','w','Position',[160 160 900 380]);
subplot(1,2,1); plot(qList, psnrQ,'-o','LineWidth',1.4); grid on;
xlabel('JPEG quality'); ylabel('PSNR (dB)'); title('PSNR vs JPEG quality');
subplot(1,2,2); plot(qList, crQ,'-s','LineWidth',1.4); grid on;
xlabel('JPEG quality'); ylabel('Compression ratio'); title('Compression ratio vs JPEG quality');

%% ---------------- 7. OUTPUT PARAMETER TABLE ---------------------------
fprintf('\n=============== OUTPUT PARAMETER TABLE ===============\n');
fprintf('%-28s %10s %10s %12s\n','Image','MSE','PSNR (dB)','MSE/MAXI^2');
fprintf('%-28s %10.2f %10.2f %12.6f\n','Median filtered',    MSE_med, PSNR_med, NIE_med);
fprintf('%-28s %10.2f %10.2f %12.6f\n','Enhanced (imadjust)',MSE_enh, PSNR_enh, NIE_enh);
fprintf('%-28s %10.2f %10.2f %12.6f\n','JPEG Q=50 reconstructed', MSE_rec, PSNR_rec, NIE_rec);

fprintf('\n--- Compression results (JPEG Q = %d) ---\n', jpegQuality);
fprintf('Original size    : %d bytes\n', origBytes);
fprintf('Compressed size  : %d bytes\n', compBytes);
fprintf('Compression ratio: %.2f : 1\n', CR);
fprintf('Bits per pixel   : %.3f bpp\n', bpp);
fprintf('SSIM (JPEG vs original): %.4f\n', SSIM_rec);

fprintf('\n--- Edge detection (Canny [%.1f %.1f]) ---\n', cannyTh);
fprintf('Edge pixels - original image   : %d\n', edgePix(1));
fprintf('Edge pixels - enhanced image   : %d\n', edgePix(2));
fprintf('Edge pixels - JPEG reconstructed: %d\n', edgePix(3));