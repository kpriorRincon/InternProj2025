classdef SMIQSegmented
    % 160MHz I/Q capture (SM200B/SM435B only)

    % The SM200B/SM435B has a 160MHz I/Q capture bandwidth for captures up
    % to 2 seconds of I/Q with complex triggering available. This class
    % exposes these captures in a simple interface. You can specify
    % immediate captures or triggered captures with video, external, and
    % frequency mask trigger types.
    
    properties (Access = private)
        DeviceHandle = -1;        
    end
    
    properties
        % Serial number of the device. Valid after construction.
        SerialNumber = -1;
        % Center freq in Hz of the I/Q capture
        CenterFrequency = 1.0e9;
        % Expected input power level (dBm)
        RefLevel = -20;
        % Specifies capture size, total capture size = pre + post
        PreTrigger = 0;
        PostTrigger = 1024;
        % Available trigger types = 'imm', 'video', 'ext', 'fmt'
        TriggerType = 'imm';
        % If video trigger enabled, this is the trigger level in dBm
        VideoTriggerLevel = -20;
        % Available trigger edge for video or ext trigger.
        % Possible values are 'rising' or 'falling'
        TriggerEdge = 'rising';
        % Size of the FMT FFT, must be a power of two from 512 to 16384
        FMTSize = 1024;
        % A list of frequencies specifying the frequency mask. 
        % Should be increasing in value between -100MHz and +100MHz. Must
        % be the same length as FMTAmplitudes.
        FMTFrequencies = [-100e6, 100e6];
        % A list of amplitudes specifying the frequency mask. 
        % Has a 1 to 1 correspondence to the FMTFrequency array. Must be
        % the same length as FMTFrequencies.
        FMTAmplitudes = [-80.0, -80.0];
        % Length of time in seconds to wait for a trigger
        TimeoutPeriod = 1.0;
        % Specify the format of the I/Q data returned.
        % 'non-interleaved' = matlab complex representation
        % 'interleaved' = interleave I/Q array, column vector
        OutputFormat = 'non-interleaved';
        % Last status reported by API
        Status = 0;
    end
    
    methods
        function obj = SMIQSegmented()
            % Establish connection with an SM200B/SM435B device.
            % Try to open the SM API
            % Throws error if not correct device.
            % Check Status to verify the device connected successfully.
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
            
            % Open device
            handleptr = libpointer('int32Ptr', -1);
            obj.Status = calllib('sm_api', 'smOpenDevice', handleptr);
            if(obj.Status < 0)
                obj.DeviceHandle = -1;
                return;
            else
                obj.DeviceHandle = handleptr.value;
            end
            
            % Get serial number
            serialptr = libpointer('int32Ptr', -1);
            devicetypeptr = libpointer('SmDeviceType', 0);            
            obj.Status = calllib('sm_api', 'smGetDeviceInfo', obj.DeviceHandle, ...
                devicetypeptr, serialptr);
            if(obj.Status >= 0)
                obj.SerialNumber = serialptr.value;
            end
            
            if ~strcmp(devicetypeptr.value, 'smDeviceTypeSM200B') && ...
                    ~strcmp(devicetypeptr.value, 'smDeviceTypeSM435B')
                error('Invalid device type, must be SM200B or SM435B');
            end
        end % constructor
        
        function delete(obj)
            % Free resources associated with a device.
            calllib('sm_api', 'smCloseDevice', obj.DeviceHandle);
        end % destructor
        
        function init(obj)
            % Initialize the I/Q capture using the properties            
            if strcmp(obj.TriggerEdge, 'rising')
                triggerEdgeEnum = 0;
            elseif strcmp(obj.TriggerEdge, 'falling')
                triggerEdgeEnum = 1;
            else
                error('Invalid TriggerEdge value');
            end
            
            if strcmp(obj.TriggerType, 'imm')
                triggerEnum = 0;
            elseif strcmp(obj.TriggerType, 'video')
                triggerEnum = 1;
                calllib('sm_api', 'smSetSegIQVideoTrigger', ...
                    obj.DeviceHandle, obj.VideoTriggerLevel, triggerEdgeEnum);
            elseif strcmp(obj.TriggerType, 'ext')
                triggerEnum = 2;
                calllib('sm_api', 'smSetSegIQExtTrigger', ...
                    obj.DeviceHandle, triggerEdgeEnum);
            elseif strcmp(obj.TriggerType, 'fmt')
                triggerEnum = 3;
                fmtSize = length(obj.FMTFrequencies);
                if fmtSize == 0
                    error('Invalid FMT size');
                end
                if fmtSize ~= length(obj.FMTAmplitudes)
                    error('FMT array sizes must match');
                end
                fmtFreqPtr = libpointer('doublePtr', obj.FMTFrequencies);
                fmtAmplPtr = libpointer('doublePtr', obj.FMTAmplitudes);
                calllib('sm_api', 'smSetSegIQFMTParams', obj.DeviceHandle, ...
                    obj.FMTSize, fmtFreqPtr, fmtAmplPtr, fmtSize);
            else
                error('Invalid TriggerType value');
            end
                        
            calllib('sm_api', 'smSetSegIQCenterFreq', obj.DeviceHandle, obj.CenterFrequency);
            calllib('sm_api', 'smSetRefLevel', obj.DeviceHandle, obj.RefLevel);
            calllib('sm_api', 'smSetSegIQSegmentCount', obj.DeviceHandle, 1);
            calllib('sm_api', 'smSetSegIQSegment', obj.DeviceHandle, 0, triggerEnum, ...
                obj.PreTrigger, obj.PostTrigger, obj.TimeoutPeriod);
            % 
            obj.Status = calllib('sm_api', 'smConfigure', obj.DeviceHandle, 5);
        end % init
        
        function [iq, timedOut] = recv(obj)
            % Perform a single capture. Device must be initialized.
            % iq is a complex array.
            % timedOut is 1 if a triggered acquisition is requested and a
            % trigger was not found, 0 otherwise.
            capturelen = obj.PreTrigger + obj.PostTrigger;
            iqarrayptr = libpointer('singlePtr', zeros(capturelen*2, 1));
            timedoutptr = libpointer('SmBool', 0);

            obj.Status = calllib('sm_api', 'smSegIQCaptureFull', obj.DeviceHandle, ...
                0, iqarrayptr, 0, capturelen, 0, timedoutptr);
             
            if strcmp(timedoutptr.value, 'smTrue')
                timedOut = true;
            elseif strcmp(timedoutptr.value, 'smFalse')
                timedOut = false;
            end

            if strcmp(obj.OutputFormat, 'non-interleaved')
                % Return results to MATLAB complex representation
                iq = reshape(iqarrayptr.Value, 2, capturelen);
                iq = iq(1,:) + 1i * iq(2,:);
                iq = transpose(iq);
            elseif strcmp(obj.OutputFormat, 'interleaved')
                % Return as interleaved I/Q pairs, column vector
                iq = iqarrayptr.Value;
            else 
                error('Invalid OutputFormat value');
            end            
        end % recv
        
        function str = getstatusstring(obj)
            % Returns a string [1xN] char.
            % Can be called after any class function to determing the success
            % of the function.
            str = calllib('sm_api', 'smGetErrorString', obj.Status);
        end % getstatusstring
    end % methods
end

