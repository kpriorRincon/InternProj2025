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

% Query device info
[status, serial] = bbgetserialnumber(handle);
[status, temp, voltage, current] = bbgetdevicediagnostics(handle);

% Done with device
bbclosedevice(handle);

fprintf('Serial Number = %d\n', serial);
fprintf('Temperature = %d Celsius\n', temp);
fprintf('Voltage = %d V\n', voltage);
fprintf('Current = %d mA\n', current);

