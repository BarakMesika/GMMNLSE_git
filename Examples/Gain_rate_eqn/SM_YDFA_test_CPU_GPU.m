% This code runs the single-mode Yb-doped fiber amplifier with the gain 
% rate equation and compare it with the SM Gaussian gain model.

clearvars; close all;

addpath('../..');
sim.cuda_dir_path = '../../cuda';

%% Gain info
gain_rate_eqn.cross_section_filename = 'Liekki Yb_AV_20160530.txt';
gain_rate_eqn.saved_mat_filename = 'SM_YDFA'; % the files used to temporarily store the data during iterations if necessary
gain_rate_eqn.reuse_data = false; % For a ring or linear cavity, the pulse will enter a steady state eventually.
                                  % If reusing the pump and ASE data from the previous roundtrip, the convergence can be much faster, especially for counterpumping.
gain_rate_eqn.linear_oscillator_model = 0; % For a linear oscillator, there are pulses from both directions simultaneously, which will deplete the gain;
                                           % therefore , the backward-propagating pulses need to be taken into account.
gain_rate_eqn.core_diameter = 6; % um
gain_rate_eqn.cladding_diameter = 125; % um
gain_rate_eqn.core_NA = 0.12;
gain_rate_eqn.absorption_wavelength_to_get_N_total = 920; % nm
gain_rate_eqn.absorption_to_get_N_total = 0.55; % dB/m
gain_rate_eqn.pump_wavelength = 976; % nm
gain_rate_eqn.copump_power = 0.4; % W
gain_rate_eqn.counterpump_power = 0; % W
gain_rate_eqn.midx = 1; % the mode index
gain_rate_eqn.mode_volume = 1; % the total number of available spatial modes in the fiber
gain_rate_eqn.t_rep = 1/15e6; % assume 15MHz here; s; the time required to finish a roundtrip (the inverse repetition rate of the pulse)
                             % This gain model solves the gain of the fiber under the steady-state condition; therefore, the repetition rate must be high compared to the lifetime of the doped ions.
gain_rate_eqn.tau = 840e-6; % lifetime of Yb in F_(5/2) state (Paschotta et al., "Lifetme quenching in Yb-doped fibers"); in "s"
gain_rate_eqn.export_N2 = false; % whether to export N2, the ion density in the upper state or not
gain_rate_eqn.ignore_ASE = true;
gain_rate_eqn.max_iterations = 10; % For counterpumping or considering ASE, iterations are required.
gain_rate_eqn.tol = 1e-5; % the tolerance for the iteration
gain_rate_eqn.allow_coupling_to_zero_fields = false; % set this to false all the time for now
gain_rate_eqn.verbose = true; % show the information(final pulse energy) during iterations of computing the gain

%% Field and simulation parameters
time_window = 50; % ps
N = 2^13; % the number of time points
dt = time_window/N;
t = (-N/2:N/2-1)'*dt; % ps

fiber.L0 = 2; % m; the length of the gain fiber
fiber.MFD = 6; % um; the mode-field diameter of the gain fiber
fiber.t_rep = gain_rate_eqn.t_rep; % for calculating the saturation intensity
save_num = 100; % the number of saved data
sim.save_period = fiber.L0/save_num;

sim.lambda0 = 1030e-9; % central wavelength; in "m"

% SM Gaussian gain
sim_SMgain = sim;
sim_SMgain.gain_model = 1;
fiber_SMgain = fiber;
fiber_SMgain.db_gain = 100; % the small-signal dB gain
fiber_SMgain.saturation_intensity = 30; % J/m^2
fiber_SMgain.saturation_energy = fiber_SMgain.saturation_intensity*(pi*(fiber.MFD*1e-6/2)^2)*1e9; % nJ
[fiber_SMgain,sim_SMgain] = load_default_GMMNLSE_propagate(fiber_SMgain,sim_SMgain);

% Rate-equation gain
sim_rategain = sim;
sim_rategain.gain_model = 4;
[fiber_rategain,sim_rategain] = load_default_GMMNLSE_propagate(fiber,sim_rategain);

fiber = {fiber_SMgain fiber_rategain};
sim = {sim_SMgain sim_rategain};

%% Initial pulse
total_energy = 0.01; % nJ
tfwhm = 1; % ps
input_field = build_MMgaussian(tfwhm, time_window, total_energy, 1, N);

%% Gain parameters
% We need some parameters of gain before computations.
f = ifftshift( (-N/2:N/2-1)'/N/dt + sim{1}.f0 ); % in the order of "omegas" in the "GMMNLSE_propagate.m"
c = 299792.458; % nm/ps;
lambda = c./f; % nm
[gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total] = gain_info( sim{2},gain_rate_eqn,lambda );
gain_param = {gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total}; % send this into GMMNLSE_propagate function

%% Propagation
if sim_SMgain.gpu_yes
    reset(gpuDevice);
end
t_end = zeros(1,2);
model_name = {'SM gain','rate-eqn gain'};
output_field = cell(1,2);

% GPU
disp('GPU:');
profile on
for i = 1:2
    t_start = tic;
    output_field{i} = GMMNLSE_propagate(fiber{i},input_field,sim{i},gain_param);
    t_end(i) = toc(t_start);
    t_spent = datevec(t_end(i)/3600/24);
    fprintf('Running time for %s: %2u:%3.1f\n',model_name{i},t_spent(5),t_spent(6));
end
profile off
profsave(profile('info'),'profile_results_GPU');

% CPU
disp('CPU:');
profile on
for i = 1:2
    sim{i}.gpu_yes = false;
    
    t_start = tic;
    output_field{i} = GMMNLSE_propagate(fiber{i},input_field,sim{i},gain_param);
    t_end(i) = toc(t_start);
    t_spent = datevec(t_end(i)/3600/24);
    fprintf('Running time for %s: %2u:%3.1f\n',model_name{i},t_spent(5),t_spent(6));
end
profile off
profsave(profile('info'),'profile_results_CPU');

%% Plot results
energy_SMgain   = permute(sum(trapz(abs(output_field{1}.fields).^2),2)*dt/1e3,[3 2 1]);
energy_rategain = permute(sum(trapz(abs(output_field{2}.fields).^2),2)*dt/1e3,[3 2 1]);

% Energy
distance = (0:save_num)*sim{1}.save_period;
figure;
plot(distance,[energy_SMgain energy_rategain]);
legend('SMgain','rategain');
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
xlabel('Time (ps)');
ylabel('Intensity (W)');
title('The final output field of YDFA (SM gain)');

% Spectrum
subplot(2,1,2);
plot(lambda,abs(fftshift(ifft(output_field{1}.fields(:,:,end)),1)).^2.*factor);
xlabel('Wavelength (nm)');
ylabel('Intensity (a.u.)');
title('The final output spectrum of YDFA (SM gain)');
xlim([1010 1080]);

% -------------------------------------------------------------------------
% rategain
% -------------------------------------------------------------------------
% Field
figure;
subplot(2,1,1);
plot(t,abs(output_field{2}.fields(:,:,end)).^2);
xlabel('Time (ps)');
ylabel('Intensity (W)');
title('The final output field of YDFA (rate-equation gain)');

% Spectrum
subplot(2,1,2);
plot(lambda,abs(fftshift(ifft(output_field{2}.fields(:,:,end)),1)).^2.*factor);
xlabel('Wavelength (nm)');
ylabel('Intensity (a.u.)');
title('The final output spectrum of YDFA (rate-equation gain)');
xlim([1010 1080]);

%% Save results
save('SM_YDFA.mat');