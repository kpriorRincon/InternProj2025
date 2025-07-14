% This example allows you to verify that the receiver is able to sustain a
% desired sample rate and throughput for a given setup. For testing full
% rate contiguous acquisition. Single burst IQ acquisition is not subject
% to data loss or similar.

% Pass either 'USB' or 'Networked'
sm = SMIQStreamer('USB');
fprintf('SN = %d\n', sm.SerialNumber);

sm.CenterFrequency = 2.440e9;
sm.DecimationFactor = 8;
sm.SoftwareFilterEnabled = false;
sm.Bandwidth = sm.samplerate() * 0.8;
sm.RefLevel = 0.0;
sm.PreselectorEnabled = 0;
sm.OutputFormat = 'interleaved';

% Modify these to fit your requirements
% number of seconds to measure
measduration = 5;
samplesPerIter = 262144;

iters = (measduration * sm.samplerate()) / samplesPerIter;

sm.start();

% Main loop
timestart = tic;
for i = 1 : iters
    iq = sm.recv(samplesPerIter, false);
end
timestop = toc(timestart);

sm.stop();

% Report
samplesreceived = iters * samplesPerIter;
fprintf('Requested sample rate = %f Msps\n', 1.0e-6 * sm.samplerate());
fprintf('%d M samples in %f seconds\n', 1.0e-6 * samplesreceived, timestop);
fprintf('Measured sample rate = %f Msps\n', 1.0e-6 * samplesreceived / timestop);

%delete(sm);
