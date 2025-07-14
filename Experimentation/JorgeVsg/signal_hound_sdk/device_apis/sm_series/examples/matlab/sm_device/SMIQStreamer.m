classdef SMIQStreamer
    % This SMIQStreamer object allows you to interface, configure, and
    % stream I/Q data from the SM200 and SM435 spectrum analyzers.
    %
    % Streaming is supported for all SM devices, but it has been shown
    % that the 100MS/s and 200MS/s rates cannot be sustained in matlab for
    % the SM200C. Single acquisitions at those rates still work. See recv()
    % for more information.
    %
    % SMIQStreamer methods:
    % SMIQStreamer - Enumerate and establish connection with an SM
    %   spectrum analyzer. Check the Status property to determine if a
    %   device connection was established. If this is the first time
    %   connecting to an SM since it was power cycled, expect a
    %   20 second connection time, after which the connection time is
    %   reduced to roughly 2 seconds. Any active or previously opened
    %   devices found are first closed before attempting to establish
    %   connection. Because of this, only one device can be managed with
    %   this object, creating two objects does not result in two
    %   devices being managed.
    % delete - Closes the device through the device DLL. If the destructor
    %   is not called before attempting to reconnect to the same device,
    %   similar functionality is performed in the constructor. Therefore
    %   you only need to ensure the destructor is called if you wish to
    %   either reclaim the resources owned by the device DLL or you wish to
    %   interact with the device in another application/thread/etc.
    % samplerate - Retrieve the sample rate of the receiver based on the
    %   current value of the DecimationFactor property.
    % start - Start the IQ data stream using the current value of the
    %   properties. Will stop an active IQ data stream if it wasn't
    %   previously stopped.
    % stop - Stop and active IQ data stream. Since the start() function
    %   will call stop() if the user has not done so, a main reason you
    %   might want to call stop is to reclaim resources owned by the device
    %   DLL (potentially several hundred MB of memory) and stop active
    %   processing which could be considerable based on the settings.
    % recv - Perform an IQ data acquisition on an active IQ data stream.
    %   Acquire 'n' complex samples.
    %   If purge is set to true, discard any buffered leftover data.
    %   If purge is set to false retrieve the next contiguous 'n' samples.
    %   If continuous streaming is desired, this function should not be
    %   called with small sizes of 'n' as this function does contain some
    %   overhead. An acceptable size of 'n' might be 1/100th of the sample
    %   rate and above. For single block acquisitions, any size of 'n' is
    %   fine.
    % getstatusstring - Returns a string [1xN] char.
    %   Can be called after any class function to determing the success
    %   of the function.
    %
    % SMIQStreamer properties:
    % SerialNumber - Set by the constructor. If failed to initialize
    %   properly, will be set to -1.
    % OutputFormat - The IQ data is provided as single precision complex
    %   values. Set to 'interleaved' (fastest) to receive a single column
    %   vector where the values are interleaved complex pairs, [Re1; Im1;
    %   Re2; Im2; ...]. Set to 'non-interleaved' (slow) to retreive complex
    %   values as MATLAB defines them ([Nx2] matrix). Warning: the
    %   'non-interleaved' return format is significantly slower than the
    %   'interleaved' format and may have difficulties maintaining
    %   throughput at the highest sample rates.
    % CenterFrequency - Center frequency of the IQ data stream in Hz. Set
    %   directly.
    % DecimationFactor - Must be set to a power of two between [1, 4096].
    %   This value directly divides the base sample rate of the SM200 of
    %   50MS/s.
    % Bandwidth - User selectable bandwidth in Hz. When the decimation
    %   factor is greater than 8 or when the software filter is enabled,
    %   this value controls the applied digital filter bandwidth. It will
    %   be clamped to reasonable values for the given selected sample rate.
    %   Specified in Hz.
    % PreselectorEnabled - Enable preselector filters below 645MHz. See the
    %   product manual for a list of filter bandwidths. Set to true or
    %   false.
    % SoftwareFilterEnabled - If true and decimation <= 8, then an
    %   additional software filter is performed on the data at the
    %   specified bandwidth. Set to true or false.
    % RefLevel - Set to any level above the expected input power (dBm).
    % Status - Reports the status of the last executed function. Use the
    %   getstatusstring() function to retrieve a human readable string of
    %   the reported status.
    
    properties
        % Internal use only.
        DeviceHandle = -1
        % Serial number of the device. Valid after construction.
        SerialNumber = -1
        % Center frequency of the IQ data stream. (In Hz)
        CenterFrequency = 1.0e9
        % Sample rate = 50.0e6 / DecimationFactor. Powers of 2 only.
        DecimationFactor = 1
        % Enable the digital software filter.
        SoftwareFilterEnabled = false
        % IQ data bandwidth. (3dB point)
        Bandwidth = 40.0e6
        % Expected input power level. (dBm)
        RefLevel = 0.0
        % Enable preselector filters. (below 645MHz)
        PreselectorEnabled = false
        % Specify the output IQ data format.
        OutputFormat = 'interleaved'
        % Last status reported by API.
        Status = 0
        % Used to calculate sample rate
        DeviceType = 'USB'
    end
    methods
        % Constructor
        % loads library, closes any open devices, and attempts to open a
        % new device. Specify either 'USB' or 'Networked' as the parameter.
        % Defaults to 'USB'. When 'Networked' uses the default network
        % configuration values.
        function obj = SMIQStreamer(deviceType)
            if nargin == 0
                deviceType = 'USB';
            else
                if ~strcmp(deviceType, 'USB')
                    if ~strcmp(deviceType, 'Networked')
                        error 'Invalid device type';
                    end
                end
            end
            
            % Establish connection with an SM device.
            obj.DeviceType = deviceType;
            
            % Try to open the SM API
            if not(libisloaded('sm_api'))
                loadlibrary('sm_api', 'sm_api.h');
                % Test the library was loaded before continuing
                if(not(libisloaded('sm_api')))
                    return;
                end
                
            else
                % Libary is already loaded. As a precaution, close all
                % possible handles It's possible the last execution ended
                % without properly closing the device, so just free all
                % resources and potentially opened devices. If at some
                % point you want to target multiple devices with this
                % class, you will need to either remove this logic or write
                % more targetted logic, i.e. close only the device you are
                % interested in, or maybe, look to see if a device is
                % already open and just return that?
                for i = (0 : 7)
                    calllib('sm_api', 'smCloseDevice', i);
                end
            end
            
            % Create handle pointer
            handleptr = libpointer('int32Ptr', -1);
            
            if(strcmp(deviceType, 'USB'))
                % open USB device
                obj.Status = calllib('sm_api', 'smOpenDevice', handleptr);
            else
                % open networked device
                hostAddr = '0.0.0.0';
                deviceAddr = '192.168.2.10';
                obj.Status = calllib('sm_api', 'smOpenNetworkedDevice', ...
                    handleptr, hostAddr, deviceAddr, 51665);
            end
            
            if(obj.Status < 0)
                obj.DeviceHandle = -1;
                return;
            else
                obj.DeviceHandle = handleptr.value;
            end
            
            % Get serial number
            serialptr = libpointer('int32Ptr', -1);
            
            obj.Status = calllib('sm_api', 'smGetDeviceInfo', obj.DeviceHandle, 0, serialptr);
            if(obj.Status >= 0)
                obj.SerialNumber = serialptr.value;
            end
        end % constructor
        
        function delete(obj)
            % Free resources associated with a device.
            calllib('sm_api', 'smCloseDevice', obj.DeviceHandle);
        end
        
        function sr = samplerate(obj)
            % Retrieve the sample rate of the device. (In S/s)
            if(strcmp(obj.DeviceType, 'USB'))
                sr = 50.0e6 / obj.DecimationFactor;
            else
                sr = 200.0e6 / obj.DecimationFactor;
            end
        end
        
        function start(obj)
            % Initialize the IQ data stream.
            
            % Configuration
            obj.Status = calllib('sm_api', 'smSetIQCenterFreq', obj.DeviceHandle, obj.CenterFrequency);
            obj.Status = calllib('sm_api', 'smSetIQBaseSampleRate', obj.DeviceHandle, 0);
            calllib('sm_api', 'smSetIQSampleRate', obj.DeviceHandle, obj.DecimationFactor);
            calllib('sm_api', 'smSetIQBandwidth', obj.DeviceHandle, int32(obj.SoftwareFilterEnabled),  obj.Bandwidth);
            calllib('sm_api', 'smSetRefLevel', obj.DeviceHandle, obj.RefLevel);
            calllib('sm_api', 'smSetPreselector', obj.DeviceHandle, int32(obj.PreselectorEnabled));
            
            % Start IQ
            obj.Status = calllib('sm_api', 'smConfigure', obj.DeviceHandle, 3);
        end
        
        function stop(obj)
            % Halt the IQ data stream.
            calllib('sm_api', 'smAbort', obj.DeviceHandle);
        end
        
        function iq = recv(obj, n, purge)
            % Perform an IQ data acquisition on an active IQ data stream.
            % This function cannot sustain the 100 and 200MS/s rates of the
            % SM200C. Creating the libpointer and retreiving the data from
            % the libpointer via ptr.Value are the bottlenecks. That said,
            % setting purge to true and retrieving a block of I/Q data at
            % the 100 and 200MS/s rates is possible.
            
            iqarrayptr = libpointer('singlePtr', zeros(n*2, 1));
            
            obj.Status = calllib('sm_api', 'smGetIQ', obj.DeviceHandle, ...
                iqarrayptr, n, 0, 0, 0, int32(purge), 0, 0);
            
            if strcmp(obj.OutputFormat, 'non-interleaved')
                % Return results to MATLAB complex representation
                iq = reshape(iqarrayptr.Value, 2, n);
                iq = iq(1,:) + 1i * iq(2,:);
                iq = transpose(iq);
            else
                % Return as interleaved I/Q pairs, column vector
                iq = iqarrayptr.Value;
            end
        end
        
        function str = getstatusstring(obj)
            % Returns a string [1xN] char.
            % Can be called after any class function to determing the success
            % of the function.
            str = calllib('sm_api', 'smGetErrorString', obj.Status);
        end
    end % methods
end