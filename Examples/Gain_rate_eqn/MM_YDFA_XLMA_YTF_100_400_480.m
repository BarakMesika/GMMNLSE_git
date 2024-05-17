% This code runs the multimode Yb-doped fiber amplifier with the gain 
% rate equation.
%

clearvars; close all;

%% Add the folders of multimode files and others
addpath('../..'); % add where many GMMNLSE-related functions like  "GMMNLSE_propagate" is
sim.cuda_dir_path = '../../cuda'; % add cuda folder
fiber.MM_folder = '../../Fibers/XLMA-YTF-100-400-480/';
fiber.betas_filename = 'betas.mat';
fiber.S_tensors_filename = 'S_tensors_50modes.mat';

%% Gain info
gain_rate_eqn.MM_folder = fiber.MM_folder; % specify the folder with the eigenmode profiles
gain_rate_eqn.cross_section_filename = 'Liekki Yb_AV_20160530.txt';
gain_rate_eqn.saved_mat_filename = 'MM_YDFA_strong_waveguide_rate_eqn'; % the files used to temporarily store the data during iterations if necessary
gain_rate_eqn.reuse_data = false; % For a ring or linear cavity, the pulse will enter a steady state eventually.
                                  % If reusing the pump and ASE data from the previous roundtrip, the convergence can be much faster, especially for counterpumping.
gain_rate_eqn.linear_oscillator_model = 0; % For a linear oscillator, there are pulses from both directions simultaneously, which will deplete the gain;
                                           % therefore , the backward-propagating pulses need to be taken into account.
gain_rate_eqn.core_diameter = 96; % um
gain_rate_eqn.cladding_diameter = 400; % um
gain_rate_eqn.core_NA = 0.11; % in fact, this is only used in single-mode
gain_rate_eqn.absorption_wavelength_to_get_N_total = 915; % nm
gain_rate_eqn.absorption_to_get_N_total = 7.5; % dB/m
gain_rate_eqn.pump_wavelength = 976; % nm
gain_rate_eqn.copump_power = 50; % W
gain_rate_eqn.counterpump_power = 0; % W
gain_rate_eqn.midx = 1:50; % the mode index
gain_rate_eqn.mode_volume = 50; % the total number of available spatial modes in the fiber
gain_rate_eqn.downsampling_factor = 1; % downsample the eigenmode profiles to run faster
gain_rate_eqn.t_rep = 1/15e6; % assume 15MHz here; s; the time required to finish a roundtrip (the inverse repetition rate of the pulse)
                             % This gain model solves the gain of the fiber under the steady-state condition; therefore, the repetition rate must be high compared to the lifetime of the doped ions.
gain_rate_eqn.tau = 840e-6; % lifetime of Yb in F_(5/2) state (Paschotta et al., "Lifetme quenching in Yb-doped fibers"); in "s"
gain_rate_eqn.export_N2 = true; % whether to export N2, the ion density in the upper state or not
gain_rate_eqn.ignore_ASE = true;
gain_rate_eqn.max_iterations = 10; % For counterpumping or considering ASE, iterations are required.
gain_rate_eqn.tol = 1e-5; % the tolerance for the iteration
gain_rate_eqn.allow_coupling_to_zero_fields = false; % set this to false all the time for now
gain_rate_eqn.verbose = true; % show the information(final pulse energy) during iterations of computing the gain

%% Field and simulation parameters
time_window = 50; % ps
N = 2^14; % the number of time points
dt = time_window/N;
t = (-N/2:N/2-1)'*dt; % ps

fiber.L0 = 1.4; % m; the length of the gain fiber
save_num = 200;
sim.save_period = fiber.L0/save_num;
sim.single_yes = false;
sim.adaptive_deltaZ.model = 0; % turn adaptive-step off
sim.step_method = "RK4IP"; % use "MPA" instead of the default "RK4IP"
sim.deltaZ = 500e-6; % m; 
sim.lambda0 = 1030e-9; % central wavelength; in "m"
sim.Raman_model = 0;
%sim.progress_bar = false;

% Rate-equation gain
sim.gain_model = 4;
[fiber,sim] = load_default_GMMNLSE_propagate(fiber,sim,'multimode');

% % just to make sure LP11a and LP11b have the same betas
% fiber.betas(:,[2,3]) = (fiber.betas(:,[2,3])+fiber.betas(:,[3,2]))/2;

%% Gain parameters
% Precompute some parameters related to the gain to save the computational time
% Check "gain_info.m" for details.
f = ifftshift( (-N/2:N/2-1)'/N/dt + sim.f0 ); % in the order of "omegas" in the "GMMNLSE_propagate.m"
c = 299792.458; % nm/ps;
lambda = c./f; % nm
[gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN] = gain_info( sim,gain_rate_eqn,lambda );
gain_param = {gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN}; % send this into GMMNLSE_propagate function

%% Initial pulse
input_field.dt = dt;
input_field.fields = zeros([size(t,1) 50]);

tmp = exp(-2.77*(t/0.6).^2);
tmp = tmp/sqrt( dt*sum(abs(tmp).^2)*1e-3 )*sqrt(5);
input_field.fields(:,1) = tmp;

% tmp = exp(-2.77*((t+7.9-1.5)/0.7).^2).*exp(+1i*613.5*t);
% tmp = fft(ifft(tmp).*exp(-1i*2*(f-363).^2));
% tmp = tmp/sqrt( dt*sum(abs(tmp).^2)*1e-3 )*sqrt(5);
% input_field.fields(:,2) = tmp;
% plot(fftshift(lambda), abs(fftshift(ifft(tmp))).^2)
% xlim([1540 1560]);grid on;
% plot(t, abs(tmp).^2)
% input_field = build_MMgaussian(tfwhm, time_window, total_energy, length(gain_rate_eqn.midx), N);
% input_field.fields(:,2:end)=0;
%% Propagation
if sim.gpu_yes
    reset(gpuDevice);
end

output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);

%% Plot results

energy_rategain = permute(sum(trapz(abs(output_field.fields).^2),2)*dt/1e3,[3 2 1]);

% Energy
distance = (0:save_num)*sim.save_period;
figure;
plot(distance,energy_rategain);
xlabel('Propagation length (m)');
ylabel('Energy (nJ)');
title('Energy');

c = 299792458e-12; % m/ps
f = (-N/2:N/2-1)'/N/dt+c/sim.lambda0;
lambda = c./f*1e9;

c = 299792.458; % nm/ps
factor = c./lambda.^2; % change the spectrum from frequency domain into wavelength domain

% Field
figure;
subplot(2,1,1);
plot(t,abs(output_field.fields(:,:,end)).^2);
legend('mode 1','mode 2','mode 3');
xlabel('Time (ps)');
ylabel('Intensity (W)');
title('The final output field of YDFA (rate-equation gain)');

% Spectrum
subplot(2,1,2);
plot(lambda,abs(fftshift(ifft(output_field.fields(:,:,end)),1)).^2.*factor);
legend('mode 1','mode 2','mode 3');
xlabel('Wavelength (nm)');
ylabel('Intensity (a.u.)');
title('The final output spectrum of YDFA (rate-equation gain)');
xlim([980 1250]);

% =========================================================================
figure;
plot(output_field.z,permute(trapz(abs(output_field.fields)).^2*dt/1e3,[3 2 1]));
xlabel('Propagation length (m)');
ylabel('energy (nJ)');
legend('LP_{01}','LP_{11a}','LP_{11b}');
title('rate-equation gain');

%% Save results
save('MM_YDFA.mat');
%%
for ii=1:save_num
    plot(lambda, abs(fftshift(ifft(output_field.fields(:,1,ii)))).^2)
    hold on
    plot(lambda, abs(fftshift(ifft(output_field.fields(:,2,ii)))).^2)
    plot(lambda, abs(fftshift(ifft(output_field.fields(:,3,ii)))).^2)
    pause(0.1)
    hold off
end
%%
for ii=150:200
    for imod=1:6
        subplot(2,1,1)
        plot(t, abs(output_field.fields(:,imod,ii)).^2,...
            'DisplayName', ['Mode' num2str(imod)])
        hold on
    end
    hold off;
    xlim([-20 20]);
    legend;
    title(['Z:' num2str(output_field.z(ii))])
    for imod=1:6
        subplot(2,1,2)
        plot(fftshift(lambda), abs(fftshift(ifft(output_field.fields(:,imod,ii)))).^2,...
            'DisplayName', ['M:' num2str(imod) 'E:' num2str(dt*sum(abs(output_field.fields(:,imod,ii)).^2)*1e-3)])
        hold on
    end
        legend;
        hold off;
        xlim([700 1600])
        drawnow
        
end
%%
Energy = squeeze( dt*sum(abs(output_field.fields).^2)*1e-3 );
plot(Energy(2,:))
%%    
plot(t, abs(input_field.fields(:,1)))
hold on
plot(t, abs(input_field.fields(:,2)))
plot(t, abs(input_field.fields(:,3)))
pause(0.1)
hold off