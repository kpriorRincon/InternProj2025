sm = SMIQSegmented;
fprintf('Open Status: %s\n', sm.getstatusstring());
fprintf('SN = %d\n', sm.SerialNumber);

sm.CenterFrequency = 1.0e9;
sm.RefLevel = -20;
sm.PreTrigger = 8192;
sm.PostTrigger = 8192;
sm.TriggerType = 'ext';
sm.TriggerEdge = 'rising';
sm.TimeoutPeriod = 10.0;
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