function [status, sweep, startFreq, binSize] = bbgetsweep(handle)
% bbgetsweep: Get a single sweep from a BB60 device
% The device must have already been configured for sweep measurements
%  with the bbconfiguresweep function and returned a non-error code from
%  that function.

    sweepSizePtr = libpointer('uint32Ptr', 0);
    startFreqPtr = libpointer('doublePtr', 0.0);
    binSizePtr = libpointer('doublePtr', 0.0);

    % First, get the sweep parameters
    % We need them to allocate the proper sweep size
    status = calllib('bb_api', 'bbQueryTraceInfo', handle, ...
        sweepSizePtr, binSizePtr, startFreqPtr);
    
    if (~strcmp(status, 'bbNoError'))
        fprintf('bbInitiate error: %s\n', bbgeterrorstring(status));
        fprintf('Device unsuccessfully configured\n');
        return
    end
    
    startFreq = startFreqPtr.value;
    binSize = binSizePtr.value;
  
    ss = sweepSizePtr.value;
    
    % Allocate memory for sweep
    sweepPtr = libpointer('singlePtr', zeros(sweepSizePtr.value, 1));
    sweepPtrMax = libpointer('singlePtr', zeros(sweepSizePtr.value, 1));
    
    % Get sweep, don't check status, as it might return one of several
    % warnings
    status = calllib('bb_api', 'bbFetchTrace_32f', ...
        handle, sweepSizePtr.value, sweepPtr, sweepPtrMax);
    
    sweep = sweepPtrMax.value;
  
end

