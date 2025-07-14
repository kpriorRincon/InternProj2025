function [window] = hammingwindow(length)
% Hamming window for people without signal processing toolbox
% dst[i] = 0.54 - 0.46*cos(2*M_PI*i/(len-1));
    
    x = linspace(0, length-1, length)';
    x = x.*((2.0*pi)/(length-1));
    window = 0.54 - 0.46 * cos(x);
    window = length * window./sum(window); % normalize window

end

