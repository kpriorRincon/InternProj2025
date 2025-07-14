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
downsample = 32; % Must be a power of 2
samplerate = 40.0e6 / downsample;
bandwidth = samplerate * 0.8;
centerfreq = 1.0e9;

bbconfigureiq(handle, centerfreq, maxinput, downsample, bandwidth);

% Increasing this number drastically increases calculation time
iqlen = 16384;
[status, iqdata] = bbgetiq(handle, iqlen, true); % flush filter ramp up
[status, iqdata] = bbgetiq(handle, iqlen, true);

bbclosedevice(handle);

% Calculate average power
avgpow = sum(abs(iqdata).^2)./iqlen;
avgpow = 10.0 * log10(avgpow);

% Plot CCDF
td = 10.0 * log10(abs(iqdata).^2);
td = sort(td, 'descend');

%x = linspace(avgpow, td(1), iqlen)';
x = zeros(iqlen, 1);
y = zeros(iqlen, 1);
for i = 1:iqlen
    x(i) = 20 * (i / iqlen);
    del = x(i) + avgpow;
    y(i) = sum(td > del) / iqlen;
end

semilogy(x, y);