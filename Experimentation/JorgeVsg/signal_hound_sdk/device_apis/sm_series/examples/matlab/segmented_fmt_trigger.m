sm = SMIQSegmented;
fprintf('Open Status: %s\n', sm.getstatusstring());
fprintf('SN = %d\n', sm.SerialNumber);

sm.CenterFrequency = 1.0e9;
sm.RefLevel = -20;
sm.PreTrigger = 8192;
sm.PostTrigger = 8192;
sm.TriggerType = 'fmt';
sm.FMTSize = 1024;
sm.FMTFrequencies = [-80.0e6, +80.0e6];
sm.FMTAmplitudes = [-50.0, 0.0];
sm.TimeoutPeriod = 1.0;
sm.OutputFormat = 'non-interleaved';

% Configure the receiver
sm.init();

% Get I/Q
fprintf('Looking for trigger\n');
[iq, timedOut] = sm.recv();

if(timedOut) 
    fprintf('Capture timed out\n');
    return;
end

% If not timed out, its a valid capture, plot it
plot(10*log10(abs(iq).^2));