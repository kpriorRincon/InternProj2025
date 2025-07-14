function [status] = bbclosedevice(handle)
%BBCLOSEDEVICE Summary of this function goes here
%   Detailed explanation goes here

    status = 0;
    if(libisloaded('bb_api'))
        status = calllib('bb_api', 'bbCloseDevice', handle);
    end

end

