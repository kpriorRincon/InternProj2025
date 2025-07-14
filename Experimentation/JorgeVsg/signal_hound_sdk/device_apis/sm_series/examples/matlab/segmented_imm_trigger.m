sm = SMIQSegmented;
fprintf('Open Status: %s\n', sm.getstatusstring());
fprintf('SN = %d\n', sm.SerialNumber);

sm.CenterFrequency = 1.0e9;
sm.RefLevel = -20;
sm.PreTrigger = 0;
sm.PostTrigger = 16384;
sm.TriggerType = 'imm';
sm.OutputFormat = 'non-interleaved';

sm.init();

[iq, timedOut] = sm.recv();

plot(10*log10(abs(iq).^2));