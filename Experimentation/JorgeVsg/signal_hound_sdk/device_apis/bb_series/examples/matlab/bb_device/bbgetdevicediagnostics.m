function [status, temp, voltage, current] = bbgetdevicediagnostics(handle)
% Returns temperature, voltage, and current of device
% temp, voltage, and current are returned as floating point values
    
    status = -1;
    temp = -1;
    voltage = -1;
    current = -1;
    
    % Ensure library is loaded
    if not(libisloaded('bb_api'))
        return;
    end

    tempptr = libpointer('singlePtr', 0);
    voltageptr = libpointer('singlePtr', 0);
    currentptr = libpointer('singlePtr', 0);
    
    status = calllib('bb_api', 'bbGetDeviceDiagnostics', ...
        handle, tempptr, voltageptr, currentptr);
    
    if (~strcmp(status, 'bbNoError'))
        return;
    else
        temp = tempptr.value;
        voltage = voltageptr.value;
        current = currentptr.value;
    end

end

