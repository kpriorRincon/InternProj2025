function [status, iqarray] = bbgetiq(handle, iqcount, purge)
% Returns an array of complex iq data
% The number of samples returned is equal to iqcount
% purge is either true or false, if true, any pending iq data
%  accumulated is removed before the samples are taken

    status = -1;

    if not(libisloaded('bb_api'))
        return;
    end
    
    iqarrayptr = libpointer('singlePtr', zeros(iqcount*2, 1));
    iqcountint = int32(iqcount);
   
    % Create integer '0' argument
    nullptr = int32(0);
    purgearg = int32(purge);
    
    status = calllib('bb_api', 'bbGetIQUnpacked', handle, ...
        iqarrayptr, iqcountint, nullptr, nullptr, purgearg, ...
        nullptr, nullptr, nullptr, nullptr);
    
    % Convert results to complex
    iqarray = reshape(iqarrayptr.Value, 2, iqcount);
    iqarray = iqarray(1,:) + i * iqarray(2,:);
    iqarray = transpose(iqarray);
    
end

