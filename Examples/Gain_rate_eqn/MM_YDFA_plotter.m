clearvars; close all;

% use 'MM_YDFA_strong_waveguide.mat',
%     'MM_YDFA_weak_waveguide.mat'
filename = 'MM_YDFA_strong_waveguide.mat';

load(filename);

energy_SMgain   = permute(sum(trapz(abs(output_field{1}.fields).^2),2)*dt/1e3,[3 2 1]);
energy_newgain  = permute(sum(trapz(abs(output_field{2}.fields).^2),2)*dt/1e3,[3 2 1]);
energy_rategain = permute(sum(trapz(abs(output_field{3}.fields).^2),2)*dt/1e3,[3 2 1]);

% Energy
distance = (0:save_num)*sim{1}.save_period;
figure;
plot(distance,[energy_SMgain energy_newgain energy_rategain]);
legend('SMgain','newgain','rategain');
xlabel('Propagation length (m)');
ylabel('Energy (nJ)');
title('Energy');

c = 299792458e-12; % m/ps
f = (-N/2:N/2-1)'/N/dt+c/sim{1}.lambda0;
lambda = c./f*1e9;

c = 299792.458; % nm/ps
factor = c./lambda.^2; % change the spectrum from frequency domain into wavelength domain

% -------------------------------------------------------------------------
% SMgain
% -------------------------------------------------------------------------
% Field
figure;
subplot(2,1,1);
plot(t,abs(output_field{1}.fields(:,:,end)).^2);
legend('LP_{01}','LP_{11a}','LP_{11b}');
xlabel('Time (ps)');
ylabel('Intensity (W)');
title('The final output field of YDFA (SM gain)');

% Spectrum
subplot(2,1,2);
plot(lambda,abs(fftshift(ifft(output_field{1}.fields(:,:,end)),1)).^2.*factor);
legend('LP_{01}','LP_{11a}','LP_{11b}');
xlabel('Wavelength (nm)');
ylabel('Intensity (a.u.)');
title('The final output spectrum of YDFA (SM gain)');
xlim([1010 1060]);

% -------------------------------------------------------------------------
% newgain
% -------------------------------------------------------------------------
% Field
figure;
subplot(2,1,1);
plot(t,abs(output_field{2}.fields(:,:,end)).^2);
legend('LP_{01}','LP_{11a}','LP_{11b}');
xlabel('Time (ps)');
ylabel('Intensity (W)');
title('The final output field of YDFA (new gain)');

% Spectrum
subplot(2,1,2);
plot(lambda,abs(fftshift(ifft(output_field{2}.fields(:,:,end)),1)).^2.*factor);
legend('LP_{01}','LP_{11a}','LP_{11b}');
xlabel('Wavelength (nm)');
ylabel('Intensity (a.u.)');
title('The final output spectrum of YDFA (new gain)');
xlim([1010 1060]);

% -------------------------------------------------------------------------
% rategain
% -------------------------------------------------------------------------
% Field
figure;
subplot(2,1,1);
plot(t,abs(output_field{3}.fields(:,:,end)).^2);
legend('LP_{01}','LP_{11a}','LP_{11b}');
xlabel('Time (ps)');
ylabel('Intensity (W)');
title('The final output field of YDFA (rate-equation gain)');

% Spectrum
subplot(2,1,2);
plot(lambda,abs(fftshift(ifft(output_field{3}.fields(:,:,end)),1)).^2.*factor);
legend('LP_{01}','LP_{11a}','LP_{11b}');
xlabel('Wavelength (nm)');
ylabel('Intensity (a.u.)');
title('The final output spectrum of YDFA (rate-equation gain)');
xlim([1010 1060]);

% =========================================================================
figure;
for i = 1:3
    subplot(1,3,i);
    plot(output_field{i}.z,permute(trapz(abs(output_field{i}.fields)).^2,[3 2 1]));
    xlabel('Propagation length (m)');
    ylabel('energy (nJ)');
    legend('LP_{01}','LP_{11a}','LP_{11b}');
    switch i
        case 1
            title_string = 'SM gain';
        case 2
            title_string = 'new gain';
        case 3
            title_string = 'rate-equation gain';
    end
    title(title_string);
end