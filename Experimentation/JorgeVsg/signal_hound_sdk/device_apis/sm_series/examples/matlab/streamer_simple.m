% This function illustrates using the streaming I/Q class to acquire a
% single I/Q capture and then plots it.

% Pass either 'USB' or 'Networked'
sm = SMIQStreamer('USB');
fprintf('Open Status: %s\n', sm.getstatusstring());
fprintf('SN = %d\n', sm.SerialNumber);

sm.CenterFrequency = 1.440e9;
sm.DecimationFactor = 1;
sm.SoftwareFilterEnabled = 1;
sm.Bandwidth = sm.samplerate() * 0.8;
sm.RefLevel = 0.0;
sm.PreselectorEnabled = 0;
sm.OutputFormat = 'non-interleaved';

% Start the receiver, collect IQ data, and stop the receiver
sm.start();
iq = sm.recv(1e6, true);
sm.stop();

% plot the data, AM vs time.
plot(10*log10(abs(iq).^2));