function func = calc_chirp
%CALC_CHIRP 

func.Gaussian = @Gaussian;
func.General  = @General;

end

function  [chirp,chirped_pulse] = Gaussian( duration,omega,omega0,bandwidth )
%CALC_CHIRP It calculates the chirp of a Gaussian pulse based on its
%duration and bandwidth.
%
%   Input:
%       duration:  FWHM under time domain; ps
%       bandwidth: FWHM under frequency domain; THz
%
%   Output:
%       chirp: the chirp; ps^2
%
%   The chirp is the C in "exp(i*C/2* w^2)" under frequency domain.
%   Please check "Agrawal, Ch.3.2, Gaussian pulse" for details.

t1 = duration/(2*sqrt(log(2))); % ps
w0 = bandwidth/(2*sqrt(log(2))); % THz

t0 = 1/w0; % calculate t0 from the time-bandwidth product; ps
           % For Gaussian, its time-bandwidth product equals one.

% t1 = t0*sqrt( 1+(C/t0^2)^2 ), from "Agrawal, Ch.3.2, Gaussian pulse"
chirp = t0*sqrt(t1^2-t0^2); % ps^2
chirped_pulse = fft(spectrum_amplitude.*exp(1i*chirp*(omega-omega0).^2));

end

function [chirp,chirped_pulse] = General( duration,omega,omega0,spectrum_amplitude )
%GENERAL It calculates the chirp of a spectrum for the desired pulse
%duration.

Nt = length(omega);
dt = 2*pi/(max(omega)-min(omega));
time = (ceil(-Nt/2):ceil(Nt/2-1))*dt;

find_optimal_chirp = @(C) find_fwhm( C,time,duration,omega,omega0,spectrum_amplitude );

%options = optimset('PlotFcns',@optimplotfval,'TolX',1e-20); % plot the process of optimization
options = optimset('TolX',1e-20);
min_duration = 0;
chirp = fminsearch(find_optimal_chirp,min_duration,options);
chirped_pulse = fft(spectrum_amplitude.*exp(1i*chirp*(omega-omega0).^2));

end

function difference = find_fwhm( C,time,duration,omega,omega0,spectrum_amplitude )

pulse = abs(fft(spectrum_amplitude.*exp(1i*C*(omega-omega0).^2))).^2;

% Find the FWHM with "findpeaks"
threshold = max(pulse)/1.0001;
[~,~,pulse_FWHM,~] = findpeaks(pulse,time,'MinPeakHeight',threshold,'WidthReference','halfheight','MinPeakProminence',threshold/2);

if isempty(pulse_FWHM)
    pulse_FWHM = 0;
else
    pulse_FWHM = max(pulse_FWHM);
end
difference = abs(duration - pulse_FWHM);

end