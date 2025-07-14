function [status, serialnumber] = bbgetserialnumber(handle)
% Retrieve the serial number of an open device
% Returns the serial number as an integer

    status = -1;
    serialnumber = -1;
    
    if not(libisloaded('bb_api'))
        return;
    end
    
    serialptr = libpointer('uint32Ptr', -1);
    status = calllib('bb_api', 'bbGetSerialNumber', handle, serialptr);
    if(status < 0)
        return;
    else
        serialnumber = serialptr.value;
    end
    
end

