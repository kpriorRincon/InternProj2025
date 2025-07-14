function [status, handle] = bbopendevice()
% Open a single device, and return an integer handle to the device.
% This function will close any previously open device.  
% This function will limit the number of open devices to 1, if you
%  need to interface more than 1 device simultaneously you will need
%  to rewrite this function.

    status = -1;    

    if not(libisloaded('bb_api'))
        loadlibrary('bb_api', 'bb_api.h');
        % Test the library was loaded before continuing
        if(not(libisloaded('bb_api')))
            return;
        end
        
    else
        % Libary is already loaded
        % As a precaution, close all possible handles
        for i = (0 : 7)
            calllib('bb_api', 'bbCloseDevice', i);
        end
    end
    
    handleptr = libpointer('int32Ptr', -1);
    status = calllib('bb_api', 'bbOpenDevice', handleptr);
    if(status < 0)
        handle = -1;
        return;
    else
        handle = handleptr.value;
    end

end

