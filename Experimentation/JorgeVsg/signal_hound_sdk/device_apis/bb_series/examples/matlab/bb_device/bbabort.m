function [status] = bbabort(handle)
% Set the receiver into an idle state
% Useful for freeing PC/CPU resources when not actively
%  making receiver measurements
% Does not close the device, device handle remains valid

    if not(libisloaded('bb_api'))
        return;
    end

    status = calllib('bb_api', 'bbAbort', handle);

end

