function [errorstring] = bbgeterrorstring(status)
% Convert a status code to a readable string

    % library must be loaded
    if not(libisloaded('bb_api'))
        return;
    end

    errorstring = calllib('bb_api', 'bbGetErrorString', status);
    
end

