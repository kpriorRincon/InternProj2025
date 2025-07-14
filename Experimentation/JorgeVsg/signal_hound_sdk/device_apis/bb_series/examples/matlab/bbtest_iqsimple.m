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
bandwidth = samplerate * 0.8;
centerfreq = 1.0e9;

status = bbconfigureiq(handle, centerfreq, maxinput, downsample, bandwidth);

iqlen = 1024;
% Flush IQ data filter ramp up time
[status, iqdata] = bbgetiq(handle, iqlen, true);
% Get actual data
[status, iqdata] = bbgetiq(handle, iqlen, true);

% Done with device
bbclosedevice(handle);

% Prepare and draw plots
subplot(2,1,1);
am = 10.0 * log10(abs(iqdata).^2);
plot(am);
title('AM vs Time');
subplot(2,1,2);
x = linspace(1, iqlen, iqlen);
plot(x, real(iqdata), x, imag(iqdata));
title('IQ vs. Time');
