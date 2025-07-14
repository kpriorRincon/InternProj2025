% This project allows you to verify that the receiver is
% able to sustain a desired sample rate and throughput for 
% a given setup. For testing full rate contiguous acquisition.
% Single burst IQ acquisition is not subject to data loss or similar.

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
downsample = 2;
samplerate = 40.0e6 / downsample;
% Bandwidth should not exceed 0.8 of sample rate
bandwidth = samplerate * 0.8;
if(downsample == 1) % special case at full data rate
    bandwidth = 27.0e6;
end
centerfreq = 1.0e9;

status = bbconfigureiq(handle, centerfreq, maxinput, downsample, bandwidth);

if (~strcmp(status, 'bbNoError'))
    fprintf('Through test aborted\n');
end

% Test the throughput of the receiver in Matlab
fprintf('Throughput test starting...\n');

% Modify these two values
measduration = 5; % seconds
iqlen = 262144; % iq samples per acquisition

iters = (measduration * samplerate) / iqlen;
timestart = tic;
for i = 1 : iters
    [status, iqdata] = bbgetiq(handle, iqlen, false);
end
timestop = toc(timestart);

% Done with receiver
bbclosedevice(handle);

% Report
samplesreceived = iters * iqlen;
fprintf('Requested sample rate = %f sps\n', samplerate);
fprintf('%d samples in %f seconds\n', samplesreceived, timestop);
fprintf('Measured sample rate = %f sps\n', samplesreceived / timestop);
