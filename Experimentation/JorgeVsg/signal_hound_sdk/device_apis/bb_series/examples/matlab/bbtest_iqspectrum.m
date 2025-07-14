% Retrieve an iq acquisition and plot the power spectrum of the data

% Open device, get handle to device
[status, handle] = bbopendevice();

% Check if device opened successfully
if (~strcmp(status, 'bbNoError'))
    fprintf('Error opening device\n');
    return
end

% Maximum expected input level in dBm
maxinput = -20.0; 
% How much to decimate the 40MS/s IQ data stream
% Must be a power of 2
downsample = 32;
samplerate = 40.0e6 / downsample;
% Bandwidth should not exceed 0.8 of sample rate
bandwidth = samplerate * 0.8;
centerfreq = 1.0e9;

status = bbconfigureiq(handle, centerfreq, maxinput, downsample, bandwidth);

iqlen = 1024; % should stay small'ish, less than 1M
% Flush IQ data filter ramp up time
[status, iqdata] = bbgetiq(handle, iqlen, true);
% Get actual data
[status, iqdata] = bbgetiq(handle, iqlen, true);

bbclosedevice(handle);

% Multiply IQ data by window function
windowcplx = complex(hammingwindow(iqlen), zeros(iqlen, 1));
iqdata = windowcplx.*iqdata;

freqplot = fftshift(fft(iqdata)./iqlen);
freqplot = 10.0 * log10(abs(freqplot).^2);

freqstart = centerfreq - (samplerate / 2.0);
binsize = samplerate / iqlen;
x = linspace(freqstart, freqstart + samplerate - binsize, iqlen);
x = x./1.0e6;

plot(x, freqplot);
title('Spectrum Plot');
xlabel('Frequency (MHz)');
ylabel('Amplitude(dBm)');
