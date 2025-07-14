function [status] = bbconfiguresweep(handle, config)
%BBCONFIGURESWEEP 
% Configure the device for a sweep. Only has to be called once
%  per sweep configuration. Will override a previous sweep 
%  configuration. If this function does not return a non-error
%  status code, then it did not fully configure.

    status = -1;
    
    if not(libisloaded('bb_api'))
        return;
    end
    
    status = calllib('bb_api', 'bbConfigureAcquisition', handle, config.Detector, config.Scale);
    if(~strcmp(status, 'bbNoError'))
        fprintf('bbConfigureAcquisition error: %s\n', bbgeterrorstring(status));
        fprintf('Device unsuccessfully configured\n');
        return        
    end
    
    status = calllib('bb_api', 'bbConfigureCenterSpan', handle, config.CenterFreq, config.Span);
    if (~strcmp(status, 'bbNoError'))
        fprintf('bbConfigCenterSpan error: %s\n', bbgeterrorstring(status));
        fprintf('Device unsuccessfully configured\n');
        return
    end
    
    % Atten is set to auto
    status = calllib('bb_api', 'bbConfigureRefLevel', handle, config.RefLevel);
    if (~strcmp(status, 'bbNoError'))
        fprintf('bbConfigureRefLevel error: %s\n', bbgeterrorstring(status));
        fprintf('Device unsuccessfully configured\n');
        return
    end
    
    % Gain and atten are not set, as they default to auto
    
    status = calllib('bb_api', 'bbConfigureSweepCoupling', handle, ...
        config.RBW, config.VBW, config.SweepTime, config.RBWShape, config.SpurReject);
    if(~strcmp(status, 'bbNoError'))
        fprintf('bbConfigureSweepCoupling error: %s\n', bbgeterrorstring(status));
        fprintf('Device unsuccessfully configured\n');
        return        
    end
    
     status = calllib('bb_api', 'bbConfigureProcUnits', handle, config.VideoUnits);
    if(~strcmp(status, 'bbNoError'))
        fprintf('bbConfigureProcUnits error: %s\n', bbgeterrorstring(status));
        fprintf('Device unsuccessfully configured\n');
        return        
    end
   
    % Configure the device for sweep acquisition
    % If status is bbNoError, the device will be actively
    %  making IQ measurements.
    % Query IQ data with bbgetiq()
    status = calllib('bb_api', 'bbInitiate', handle, 0, 0);
    if (~strcmp(status, 'bbNoError'))
        fprintf('bbInitiate error: %s\n', bbgeterrorstring(status));
        fprintf('Device unsuccessfully configured\n');
        return
    end
   
    
end

