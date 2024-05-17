function [Strehl_ratio,dechirped_FWHM,transform_limited_FWHM,peak_power,fig] = analyze_field( t,f,field,grating_incident_angle,grating_spacing,verbose,ASE )
%ANALYZE_FIELD It plots the field and the spectrum, as well as their
%dechirped and transform-limited counterparts.
%
% Input:
%   t: (N,1); time (ps)
%   f: (N,1); frequency (THz)
%   field: (N,?); the field to be analyzed
%                 If ? isn't 1, it'll choose the most energetic field
%   grating_incident_angle: a scalar; the incident angle of light toward the grating
%   grating_spacing: a scalar; the line spacing of the grating
%
% Optional input argument:
%   verbose: 1(true) or 0(false); whether to plot and display the results or not (default: true)
%   ASE

if nargin == 5
    verbose = true;
end
fig = [];

% Pick the strongest field only
[~,mi] = max(sum(abs(field).^2,1));
field = field(:,mi);

c = 299792.458; % nm/ps
wavelength = c./f(f>0); % nm
N = size(field,1);
dt = t(2)-t(1); % ps
factor_correct_unit = (N*dt)^2/1e3; % to make the spectrum of the correct unit "nJ/THz"
                                    % "/1e3" is to make pJ into nJ
spectrum = abs(fftshift(ifft(field),1)).^2*factor_correct_unit; % in frequency domain
spectrum = spectrum(f>0); % ignore the negative frequency part, if existing, due to a large frequency window

% -------------------------------------------------------------------------
% Unit explanation:
%   intensity = abs(field).^2;
%   energy = trapz(t,intensity) = trapz(intensity)*dt;       % pJ
%   
%   spectrum_unknown_unit = abs(fftshift(ifft(field),1)).^2;
%
%   Parseval's theorem: sum(intensity) = sum(spectrum_unknown_unit)*N;
%                       * Note that spectrum_unknown_unit is from "ifft".
%   therefore sum(intensity)*dt = sum(spectrum_unknown_unit)*N*dt
%                               = sum(spectrum_unknown_unit)*(N*dt)^2/(N*dt)
%                               = sum(spectrum_unknown_unit)*(N*dt)^2*df
%                               = sum(spectrum_f)*df
%
%   spectrum_f = spectrum_unknown_unit*(N*dt)^2;
%   energy = trapz(f,spectrum_f) = trapz(spectrum_f)*df      % pJ
%                                = trapz(spectrum_f)/(N*dt);
%
%   c = 299792.458;     % nm/ps
%   wavelength = c./f;  % nm
%   spectrum_wavelength = spectrum_f.*(c./wavelength.^2);
%   energy = -trapz(wavelength,spectrum_wavelength);         % pJ
% -------------------------------------------------------------------------

%% Plot the field and the spectrum
if verbose
    fig(1) = figure('Name','Field and Spectrum');
    fp = get(gcf,'position');
    screen_size = get(0,'ScreenSize');
    original_top = screen_size(4)-fp(2)-fp(4);
    set(gcf,'position',[fp(1) screen_size(4)-original_top-fp(4)*7/4 fp(3)*7/4 fp(4)]);
    subplot(1,2,1);
    h1 = plot(t,abs(field).^2);
    xlabel('Time (ps)'); ylabel('Intensity (W)');
    title('Field');
    ax1 = gca;
    subplot(1,2,2);
    factor = c./wavelength.^2; % change the spectrum from frequency domain into wavelength domain
    h2 = plot(wavelength,spectrum.*factor);
    xlabel('Wavelength (nm)'); ylabel('Intensity (nJ/nm)');
    xlim([min(wavelength) max(wavelength)]);
    ax2 = gca;
    set(h1,'linewidth',2); set(h2,'linewidth',2);
    set(ax1,'fontsize',16); set(ax2,'fontsize',16);
    title('Spectrum');
end

%% Dechirped and Transform-limited
% -------------------------------------------------------------------------
%{
fitted_order = 7;

% Before dechirping
if verbose
    if exist('cprintf','file')
        cprintf('*[0.3 0 0.6]','Before dechirping,\n');
    else
        disp('Before dechirping,');
        disp('----------');
    end
end
[GVD_before_dechirping,TOD_before_dechirping] = characterize_spectral_phase(f,fftshift(ifft(ifftshift(field,1)),1),fitted_order,verbose);
if verbose
    disp('===============');
end
%}
% Dechirp the pulse
[~,dechirped_FWHM,dechirped_field] = pulse_compressor('t',grating_incident_angle,feval(@(x)x(1),ifftshift(c./f,1)),t,field,grating_spacing,false);
%{
% After dechirping
if verbose
    if exist('cprintf','file')
        cprintf('*[0.3 0 0.6]','After dechirping,\n');
    else
        disp('After dechirping,');
        disp('----------');
    end
end
[GVD_after_dechirping,TOD_after_dechirping] = characterize_spectral_phase(f,fftshift(ifft(ifftshift(dechirped_field,1)),1),fitted_order,verbose);
if verbose
    disp('===============');
end
%}
% -------------------------------------------------------------------------

% Transform-limited pulse
num_interp = 5;
[transform_limited_field,t_interp,transform_limited_FWHM,pulse_FWHM] = calc_transform_limited( field,num_interp,t );

% Strehl ratio
peak_power = max(abs(dechirped_field).^2);
Strehl_ratio = peak_power/max(abs(transform_limited_field).^2);

% Plot only the central part of the tme window of the dechirped and 
% transform-limited pulse because their duration are too small compared to
% the time window
if verbose
    intensity = abs(dechirped_field).^2;
    intensity_plot = intensity;
    threshold_factor = 100;
    intensity_plot(intensity<max(intensity)/threshold_factor) = 0;
    left = find(intensity_plot~=0,1);
    right = find(intensity_plot~=0,1,'last');
    center = floor((left+right)/2);
    span_factor = 2;
    span = floor((right-left)/2)*span_factor;
    left = floor(center-span);
    right = ceil(center+span);

    fig(2) = figure('Name','Transform-limited vs. Dechirped');
    fp = get(gcf,'position');
    screen_size = get(0,'ScreenSize');
    original_top = screen_size(4)-fp(2)-fp(4);
    set(gcf,'position',[fp(1) screen_size(4)-original_top-fp(4)*5/4 fp(3:4)*5/4]);
    h3 = plot(t_interp*1e3,abs(transform_limited_field).^2/1e3);
    hold on;
    h4 = plot(t*1e3,abs(dechirped_field).^2/1e3);
    hold off;
    xlim([min(t(left:right)) max(t(left:right))]*1e3);
    xlabel('Time (fs)'); ylabel('Intensity (kW)');
    title('Transform-limited vs. Dechirped');
    legend('transform-limited','dechirped');
    set(h3,'linewidth',2); set(h4,'linewidth',2);
    set(gca,'fontsize',16);

    % Print the results
    if exist('cprintf','file')
        cprintf('blue','Pulse duration: %6.4f(fs)\n',pulse_FWHM);
        cprintf('blue','Dechirped duration: %6.4f(fs)\n',dechirped_FWHM);
        cprintf('blue','Transform-limited duration: %6.4f(fs)\n',transform_limited_FWHM);
        cprintf('red','--> Strehl ratio = %6.4f\n',Strehl_ratio);
    else
        fprintf('Pulse duration: %6.4f(fs)\n',pulse_FWHM);
        fprintf('Dechirped duration: %6.4f(fs)\n',dechirped_FWHM);
        fprintf('Transform-limited duration: %6.4f(fs)\n',transform_limited_FWHM);
        fprintf('--> Strehl ratio = %6.4f\n',Strehl_ratio);
    end
    fprintf('Peak power = %6.4f(kW)\n',peak_power/1e3);
end

% Plot the total spectrum including pulse and ASE.
if verbose && exist('ASE','var')
    fig(3) = figure('Name','Spectrum');
    %spectrum = spectrum/ASE.t_rep*1e-9; % W/THz
    %spectrum_total = spectrum.*factor + ASE.spectrum(f>0).*factor;
    spectrum_total = spectrum.*factor + ASE.spectrum(f>0).*ASE.t_rep*1e9.*factor;
    h  = plot(299792.458./f(f>0),  spectrum_total,'r'); hold on;
    h2 = plot(299792.458./f(f>0),spectrum.*factor,'b'); hold off;
    l = legend('Pulse+ASE','Pulse');
    xlabel('Wavelength (nm)'); title('Spectrum');
    %ylabel('Intensity (W/nm)');
    ylabel('Intensity (nJ/nm)');
    set(h,'linewidth',2); set(h2,'linewidth',2); set(gca,'fontsize',16); set(l,'fontsize',16);
end

end