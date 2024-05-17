% This code runs the multimode Yb-doped fiber amplifier with the gain 
% rate equation and compare it with SM Gaussian and new gain models.
%
% The parameters of the Yb-doped fiber used here is from Thorlabs
% YB1200-10-125DC, but NA is increased to 0.3 forcing higher-order modes to
% confine inside the core.
%
% Because of strong confinement, all the modes see the similar gain.
% With weak confinement, the fundamental mode sees the strongest gain. Other modes experience weak gain.
% SM and new-gain models assume all the modes see the same gain.
%
% Please compare the result of this simulation with the one from
% "MM_YDFA_weak_waveguide.m"

clearvars; close all;

%% Add the folders of multimode files and others
addpath('../..'); % add where many GMMNLSE-related functions like  "GMMNLSE_propagate" is
sim.cuda_dir_path = '../../cuda'; % add cuda folder
fiber.MM_folder = '../../Fibers/YB-10um-NA03_wavelength1030nm/';
fiber.betas_filename = 'betas.mat';
fiber.S_tensors_filename = 'S_tensors_3modes.mat';

%% Gain info
gain_rate_eqn.MM_folder = fiber.MM_folder; % specify the folder with the eigenmode profiles
gain_rate_eqn.cross_section_filename = 'Liekki Yb_AV_20160530.txt';
gain_rate_eqn.saved_mat_filename = 'MM_YDFA_strong_waveguide_rate_eqn'; % the files used to temporarily store the data during iterations if necessary
gain_rate_eqn.reuse_data = false; % For a ring or linear cavity, the pulse will enter a steady state eventually.
                                  % If reusing the pump and ASE data from the previous roundtrip, the convergence can be much faster, especially for counterpumping.
gain_rate_eqn.linear_oscillator_model = 0; % For a linear oscillator, there are pulses from both directions simultaneously, which will deplete the gain;
                                           % therefore , the backward-propagating pulses need to be taken into account.
gain_rate_eqn.core_diameter = 10; % um
gain_rate_eqn.cladding_diameter = 125; % um
gain_rate_eqn.core_NA = 0.3; % in fact, this is only used in single-mode
gain_rate_eqn.absorption_wavelength_to_get_N_total = 920; % nm
gain_rate_eqn.absorption_to_get_N_total = 1.7; % dB/m
gain_rate_eqn.pump_wavelength = 976; % nm
gain_rate_eqn.copump_power = 1; % W
gain_rate_eqn.counterpump_power = 0; % W
gain_rate_eqn.midx = 1:3; % the mode index
gain_rate_eqn.mode_volume = 3; % the total number of available spatial modes in the fiber
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
N = 2^12; % the number of time points
dt = time_window/N;
t = (-N/2:N/2-1)'*dt; % ps

fiber.L0 = 1; % m; the length of the gain fiber
fiber.MFD = 7.665; % um; the mode-field diameter
gain_Aeff = 4.6144e-11; % m^2; the Aeff of the first mode
fiber.t_rep = gain_rate_eqn.t_rep; % for calculating the saturation intensity
save_num = 100;
sim.save_period = fiber.L0/save_num;
sim.single_yes = false;

sim.lambda0 = 1030e-9; % central wavelength; in "m"
%sim.progress_bar = false;

% db_gain and saturation_intensity are adjusted to have a good bit with the rate-equation-gain model

% SM Gaussian gain
sim_SMgain = sim;
sim_SMgain.gain_model = 1;
fiber_SMgain = fiber;
fiber_SMgain.db_gain = 65.9; % the small-signal dB gain
fiber_SMgain.saturation_intensity = 37; % J/m^2
fiber_SMgain.saturation_energy = fiber_SMgain.saturation_intensity*gain_Aeff*1e9; % nJ
[fiber_SMgain,sim_SMgain] = load_default_GMMNLSE_propagate(fiber_SMgain,sim_SMgain,'multimode');

% new gain
sim_newgain = sim;
sim_newgain.gain_model = 2;
fiber_newgain = fiber;
fiber_newgain.db_gain = 65.9; % the small-signal dB gain
fiber_newgain.saturation_intensity = 37; % J/m^2
[fiber_newgain,sim_newgain] = load_default_GMMNLSE_propagate(fiber_newgain,sim_newgain,'multimode');

% Rate-equation gain
sim_rategain = sim;
sim_rategain.gain_model = 4;
[fiber_rategain,sim_rategain] = load_default_GMMNLSE_propagate(fiber,sim_rategain,'multimode');

fiber = {fiber_SMgain fiber_newgain fiber_rategain};
sim = {sim_SMgain sim_newgain sim_rategain};

for i = 1:length(fiber)
    fiber{i}.betas(:,[2,3]) = (fiber{i}.betas(:,[2,3])+fiber{i}.betas(:,[3,2]))/2;
end

%% Initial pulse
total_energy = 0.01; % nJ
tfwhm = 1; % ps
input_field = build_MMgaussian(tfwhm, time_window, total_energy, length(gain_rate_eqn.midx), N);

%% Gain parameters
% Precompute some parameters related to the gain to save the computational time
% Check "gain_info.m" for details.
f = ifftshift( (-N/2:N/2-1)'/N/dt + sim{1}.f0 ); % in the order of "omegas" in the "GMMNLSE_propagate.m"
c = 299792.458; % nm/ps;
lambda = c./f; % nm
[gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN] = gain_info( sim{2},gain_rate_eqn,lambda );
gain_param = {gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN}; % send this into GMMNLSE_propagate function
    
%% Propagation
if sim_SMgain.gpu_yes
    reset(gpuDevice);
end
t_end = zeros(1,3);
model_name = {'SM gain','new gain','rate-eqn gain'};
output_field = cell(1,3);
for i = 1:3
    t_start = tic;
    output_field{i} = GMMNLSE_propagate(fiber{i},input_field,sim{i},gain_param);
    t_end(i) = toc(t_start);
    t_spent = datevec(t_end(i)/3600/24);
    fprintf('Running time for %s: %2u:%3.1f\n',model_name{i},t_spent(5),t_spent(6));
end

%% Plot results
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
legend('mode 1','mode 2','mode 3');
xlabel('Time (ps)');
ylabel('Intensity (W)');
title('The final output field of YDFA (SM gain)');

% Spectrum
subplot(2,1,2);
plot(lambda,abs(fftshift(ifft(output_field{1}.fields(:,:,end)),1)).^2.*factor);
legend('mode 1','mode 2','mode 3');
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
legend('mode 1','mode 2','mode 3');
xlabel('Time (ps)');
ylabel('Intensity (W)');
title('The final output field of YDFA (new gain)');

% Spectrum
subplot(2,1,2);
plot(lambda,abs(fftshift(ifft(output_field{2}.fields(:,:,end)),1)).^2.*factor);
legend('mode 1','mode 2','mode 3');
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
legend('mode 1','mode 2','mode 3');
xlabel('Time (ps)');
ylabel('Intensity (W)');
title('The final output field of YDFA (rate-equation gain)');

% Spectrum
subplot(2,1,2);
plot(lambda,abs(fftshift(ifft(output_field{3}.fields(:,:,end)),1)).^2.*factor);
legend('mode 1','mode 2','mode 3');
xlabel('Wavelength (nm)');
ylabel('Intensity (a.u.)');
title('The final output spectrum of YDFA (rate-equation gain)');
xlim([1010 1060]);

% =========================================================================
figure;
for i = 1:3
    subplot(1,3,i);
    plot(output_field{i}.z,permute(trapz(abs(output_field{i}.fields)).^2*dt/1e3,[3 2 1]));
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

%% Save results
save('MM_YDFA_strong_waveguide.mat');