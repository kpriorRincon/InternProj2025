classdef BBSweepConfig
    %BBCONSTANTS Stores the properties for a full sweep configuration.
    % Provides convenience constants for several parameters to
    %  bbconfiguresweep. Should be used instead of magic numbers.
    
    % User configurable properties
    properties
        CenterFreq = 1.0e9;
        Span = 20.0e6;
        RBW = 10.0e3;
        VBW = 10.0e3;
        RefLevel = 0.0; 
        Detector = 0; % MinMax
        SweepTime = 0.001;
        RBWShape = 1; % Flattop
        SpurReject = 0; % Disabled
        VideoUnits = 2; % Power
        Scale = 0; % Log scale
    end
    
    % Used to set the above properties
    properties (Constant)
        % Scale values
        SCALE_LOG = 0;
        SCALE_LINEAR = 1;
        
        % RBWShape values
        NUTTALL = 0;
        FLATTOP = 1;
        
        % Detector values
        MIN_MAX = 0;
        AVERAGE = 1;
        
        % VideoUnits values
        LOG = 0;
        VOLTAGE = 1;
        POWER = 2;
        SAMPLE = 3;
        
        % SpurReject values
        SPUR_REJECT_DISABLED = 0;
        SPUR_REJECT_ENABLED = 1;
    end
end

