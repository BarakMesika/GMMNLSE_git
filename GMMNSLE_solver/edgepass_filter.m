function [output,fig] = edgepass_filter(type, input, f0, cutonoff_lambda, varargin)
%EDGEPASS_FILTER Apply a spectral edgefilter to a field
%
% Input:
%   type - 'lowpass' or 'highpass'; passband based on the wavelength
%   input.fields - a (N, num_modes, m) matrix with each mode's field, in the time domain
%   input.dt - the time grid spacing, in ps
%   f0 - the center frequency of simulations, in THz
%   cutonoff_lambda - the cutoff/cuton wavelength of the edgefilter, in m
%
%   Optional inputs (varargin):
%       cutonoff_slope - the slope at the cutoff/cuton wavelength (default: 0.15)
%       gaussexpo - supergaussian exponent (~exp(-t^(2*gaussexpo))) (default: 1)
%       verbose - true(1) or false(0); whether to plot the input and the output spectra (default: false)
%
% Output:
%   output.fields
%   output.rejected_fields
%   output.dt
%   fig - the figure handle of the figure if "verbose" is true

c = 2.99792458e-4; % in m/ps

if ~isstruct(input)
    error('Unlike most auxiliary functions, edgepass_filter requires that the input be a struct with at least fields and dt');
end

optargs = {0.15,1,false};
optargs(1:length(varargin)) = varargin;
[cutonoff_slope, gaussexpo, verbose] = optargs{:};

input_field = input.fields(:, :, end);
N = size(input_field, 1);
dt = input.dt;

f = f0 + ifftshift(linspace(-N/2, N/2-1, N))'/(N*dt); % in THz, in the order that the fft gives

% Calculate the filter profile in frequency space
f_0 = (2*gaussexpo-1)/cutonoff_slope*nthroot(gaussexpo/(2*gaussexpo-1),2*gaussexpo)*exp(-(2*gaussexpo-1)/(2*gaussexpo));
switch type
    case 'lowpass'
        center_f = c/cutonoff_lambda + f_0*nthroot(log(2),2*gaussexpo);
    case 'highpass'
        center_f = c/cutonoff_lambda - f_0*nthroot(log(2),2*gaussexpo);
end
mult_factor = exp(-(f-center_f).^(2*gaussexpo)/(2*f_0^(2*gaussexpo)));
switch type
    case 'lowpass'
        mult_factor(f>=center_f) = 1;
    case 'highpass'
        mult_factor(f<=center_f) = 1;
end

% Apply the filter in frequency space
output = struct('dt',input.dt,...
                'fields',fft(mult_factor.*ifft(input_field)),...
                'rejected_fields',fft((1-mult_factor).*ifft(input_field)));

if verbose
    fig = figure('Name','Filter');
    f = fftshift(f,1);
    factor_correct_unit = (N*dt)^2/1e3; % to make the spectrum of the correct unit "nJ/THz"
                                        % "/1e3" is to make pJ into nJ
    spectrum = abs(fftshift(ifft(input_field),1)).^2*factor_correct_unit;
    
    yyaxis left
    h1 = plot(f,spectrum);
    hold on;
    h2 = plot(f,spectrum.*fftshift(mult_factor,1).^2,'--');
    hold off;
    xlabel('Frequency (THz)'); ylabel('Intensity (nJ/THz)');
    title('Spectral filter');
    set(h1,'linewidth',2); set(h2,'linewidth',2);
    set(gca,'fontsize',14);
    yyaxis right
    h3 = plot(f,fftshift(mult_factor,1).^2);
    set(h3,'linewidth',2);
    drawnow;
end

end

