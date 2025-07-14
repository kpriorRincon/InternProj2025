function [apiversion] = bbgetapiversion()
% Return the API version as a string

    % Ensure library is loaded
    if not(libisloaded('bb_api'))
        loadlibrary('bb_api', 'bb_api.h');
    end

    apiversion = calllib('bb_api', 'bbGetAPIVersion');

end

