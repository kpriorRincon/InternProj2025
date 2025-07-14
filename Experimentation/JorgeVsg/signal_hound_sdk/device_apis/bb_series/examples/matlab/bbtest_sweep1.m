% This example... 
%   opens a device
%   configures a sweep
%   performs a single sweep
%   closes the device
%   plots the sweep

% Print API version
fprintf('API Version: %s\n', bbgetapiversion());
        
% Connect a BB60 device
fprintf('Opening device, est. 3 seconds\n');
[status, handle] = bbopendevice();

% Example print status string
fprintf('Open Status: %s\n', bbgeterrorstring(status));

% Check if device opened
if (~strcmp(status, 'bbNoError'))
    fprintf('Error opening device\n');
    return
end

% Create new sweep configuration 
sweepConfig = BBSweepConfig;

% Set sweep configuration
sweepConfig.CenterFreq = 1.0e9;
sweepConfig.Span = 100.0e6;
sweepConfig.RBW = 10.0e3;
sweepConfig.VBW = 10.0e3;
sweepConfig.RefLevel = 0.0;
sweepConfig.Detector = BBSweepConfig.MIN_MAX;
sweepConfig.SweepTime = 0.001;
sweepConfig.RBWShape = BBSweepConfig.FLATTOP;
sweepConfig.SpurReject = BBSweepConfig.SPUR_REJECT_DISABLED;
sweepConfig.VideoUnits = BBSweepConfig.POWER;
sweepConfig.Scale = BBSweepConfig.SCALE_LOG;

% Configure the receiver
status = bbconfiguresweep(handle, sweepConfig);
if (~strcmp(status, 'bbNoError'))
    fprintf('Error initializing device\n');
    return
end

% Perform the sweep
[status, sweep, startFreq, binSize] = bbgetsweep(handle);

% We are finished with device
bbclosedevice(handle);

% Create x axis
xaxis = linspace(0, length(sweep)-1, length(sweep));
xaxis = (xaxis * binSize + startFreq) * 1.0e-6;

% Plot sweep
plot(xaxis, sweep);
title('Sweep');
xlabel('Frequency (MHz)');
ylabel('Amplitude (dBm)');
