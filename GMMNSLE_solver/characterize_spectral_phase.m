function [quardratic_phase,cubic_phase] = characterize_spectral_phase( f,Ef,fitted_order,verbose )
%CHARATERIZE_SPECTRAL_PHASE It fits a polynomial to the spectral phase and
%finds the GVD and TOD.
%
% Input:
%   f  - (N,1); frequency (THz)
%   Ef - (N,1); the frequency-domain electric field/spectrum with both
%        amplitude and phase = fftshift(ifft(ifftshift(Et,1)),1), where Et 
%        is centered at t=0 by "ifftshift" first. 
%   fitted_order - a scalar; the order of the fitted polynomial (default: 7)
%   verbose - 1(true) or 0(false); plot the fitted curve and print the
%              results (quardratic_phase and cubic_phase) (default: false)
%
% Output:
%   quardratic_phase
%   cubic_phase
%
%   Please check "p.248, Ch.6.2.1, Applications of Nonlinear Fiber Optics"
%   for details of the convection of phase.
%       phi = phi0 + phi1*(w-w0) + 1/2*phi2*(w-w0)^2 + 1/6*phi3*(w-w0)^3 + ......

switch nargin
    case 2
        fitted_order = 7;
        verbose = false;
    case 3
        verbose = false;
end

if size(Ef,2) ~= 1
    error('characterize_spectral_phase:EfSizeError',...
          'Ef can only be (N,1) column vector.');
end

area = trapz(f,abs(Ef).^2);
center = trapz(f,f.*abs(Ef).^2)./area;
[~,center] = min(abs(f-center));

% Consider only the central (non-zero) part of the spectrum
intensity = abs(Ef).^2;
intensity_plot = intensity;
threshold_factor = 50;
intensity_plot(intensity<max(intensity)/threshold_factor) = 0;
left = find(intensity_plot~=0,1);
right = find(intensity_plot~=0,1,'last');
span = max(center-left,right-center);
left = max(1,center - span);
right = min(length(f),center + span);

f = f(left:right);
omega = 2*pi*f;
Ef = Ef(left:right);

% Fit the spectral phase with a polynomial of order "fitted_order"
[p,s,mu] = polyfit(omega,unwrap(angle(Ef)),fitted_order);
[fitted_phase,delta] = polyval(p,omega,s,mu);

quardratic_phase = p(fitted_order-1)/mu(2)^2*2*1e6; % ps^2 to fs^2
cubic_phase = p(fitted_order-2)/mu(2)^3*6*1e9; % ps^3 to fs^3

if verbose
    % Plot
    figure;
    yyaxis left
    hI = plot(f,abs(Ef).^2,'b');
    ylim([0 max(abs(Ef).^2)*1.5]);
    ylabel('Intensity (a.u.)');
    yyaxis right
    hp = plot(f,unwrap(angle(Ef)),'k');
    hold on;
    hpf = plot(f,fitted_phase,'r');
    hpfr = plot(f,fitted_phase-2*delta,'m--',f,fitted_phase+2*delta,'m--');
    hold off;
    ylabel('Phase (rad)');
    if f(1) > f(end)
        min_f = f(end);
        max_f = f(1);
    else
        min_f = f(1);
        max_f = f(end);
    end
    xlim([min_f max_f]);
    xlabel('Frequency (THz)');
    legend('Intensity','Phase','Fitted Phase','95% Prediction Interval');
    set(hI,'linewidth',2);set(hp,'linewidth',2);set(hpf,'linewidth',2);set(hpfr,'linewidth',2);
    set(gca,'fontsize',14);
    
    c = 299792.458; % nm/ps
    
    % Print the result
    fprintf('Fitted f0:      %6.4f(THz)\n',mu(1)/2/pi);
    fprintf('     = lambda0: %6.4f(nm)\n',c/(mu(1)/2/pi))
    fprintf('quardratic_phase: %8.6g(fs^2)\n',quardratic_phase);
    fprintf('cubic_phase: %8.6g(fs^3)\n',cubic_phase);
end

end

