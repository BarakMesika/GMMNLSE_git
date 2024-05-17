function [transform_limited_field,varargout] = calc_transform_limited( input_field,num_interp_points,t )
%DECHIRP_TRANSFORM_LIMITED It gives the dechirped result which is transform-limited.
%
% Input:
%   input_field: (Nt,...); the electric field in time domain (sqrt(W))
%   num_interp_points: (optional input argument)
%                      a scalar;
%                      the number of points interpolated between two original data points
%                      (default: zero)
%
%   **Because the transform-limited pulse is really narrow, 
%     the original time spacing can be too large to correctly characterize the pulse.
%     This will be a problem with calculating the transform-limited duration.
%     Therefore, increasing the number of time grid points, i.e., reduing the time spacing, is highly recommended.
%
%   t: (optional input argument)
%      (Nt,1); time (ps)
%
% Output:
%   transform_limited_field
%   t_interp: the interpolated time grid points based on "num_interp_points"
%   transform_limited_FWHM: transform-limited pulse duration (fs)
%   pulse_FWHM: current pulse duration (fs)
% =========================================================================
% "num_interp_points" and "t" are optional input arguments, but "t" is
% required to calculate "t_interp" and "transform_limited_FWHM" as output.
% =========================================================================
% Usage:
%   transform_limited_field = calc_transform_limited(input_field);
%   transform_limited_field = calc_transform_limited(input_field,num_interp_points);
%   [transform_limited_field,t_interp,transform_limited_FWHM] = calc_transform_limited(input_field,num_interp_points,t);

calc_TL_duration = false;
switch nargin
    case 1
        num_interp_points = 0;
    case 3
        calc_TL_duration = true;
end

sE = size(input_field);
Nt = sE(1);
interp_idx = linspace(1,Nt,(num_interp_points+1)*(Nt-1)+1)';

% The interpolation can't be done with only
% "interp1(input_field,interp_idx)" because an interpolation of complex
% numbers isn't correct. It should be done with two interpolations of their
% absolute values and phases.
if num_interp_points ~= 0
    abs_input_field = interp1(abs(input_field),interp_idx);
    phase_input_field = interp1(unwrap(angle(input_field)),interp_idx);
    input_field = abs_input_field.*exp(1i*phase_input_field);
end

input_freq_TL = abs(fftshift(ifft(ifftshift(input_field,1)),1));
for i = 1:prod(sE(2:end))
    input_freq_TL(abs(input_freq_TL(:,i))<0.05*max(abs(input_freq_TL(:,i))),i) = 0;
end
transform_limited_field = fftshift(fft(ifftshift(abs(input_freq_TL),1)),1);

if calc_TL_duration
    % Interpolated time
    t_interp = interp1(t,interp_idx);
    
    pulse_FWHM = zeros([1,sE(2:end)]);
    transform_limited_FWHM = zeros([1,sE(2:end)]);
    for i = 1:prod(sE(2:end))
        % Current duration
        threshold = max(abs(input_field(:,i)).^2)/1.01;
        [~,~,tmp_pulse_width,~] = findpeaks(abs(input_field(:,i)).^2,t_interp*1e3,'MinPeakHeight',threshold,'WidthReference','halfheight');
        pulse_FWHM(i) = tmp_pulse_width(1);
        
        % Transform-limited duration
        threshold = max(abs(transform_limited_field(:,i)).^2)/1.0001;
        [~,~,transform_limited_FWHM(i),~] = findpeaks(abs(transform_limited_field(:,i)).^2,t_interp*1e3,'MinPeakHeight',threshold,'WidthReference','halfheight');
    end
    
    varargout = {t_interp,transform_limited_FWHM,pulse_FWHM};
else
    varargout = {};
end

end