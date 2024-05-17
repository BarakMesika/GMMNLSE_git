% This code runs a PM amplifier with the gain rate-equation model.
%
% A small amount of y-polarized light is included in the x-polarized input
% as an imperfect input. 20dB polarization extinction ratio (PER) is used
% here.
% From the result, you can see the light stays mainly in the x-polarization
% throughout the propagation. The output still has 20dB PER.

close all; clearvars;

addpath('../../');
reset(gpuDevice);

%% Setup fiber parameters
% Please find details of all the parameters in "load_default_GMMNLSE_propagate.m".
% Only necessary parameters are set here; otherwise, default is used.
sim.lambda0 = 1030e-9; % the center wavelength of the simulation window
sim.scalar = false; % polarized fields
sim.single_yes = false; % use double precision
sim.gain_model = 4; % use gain rate-equation model
sim.gpu_yes = true; % use GPU

fiber.MFD = 6; % um; mode field diameter of the fiber
fiber.L0 = 4; % m; the length of fiber length
sim.save_period = fiber.L0/100;

betas = [8.867e6; 4.903e3; 0.0208; 33.3e-6; -27.7e-9]; % at 1030nm
c = 299792458e-12; % m/ps
beat_length = 2.7e-3; % 2.7mm at 980nm for PM980
% The betas at the other polarization can be easily calculated by 
% beta_e = n_e*w/c = (n_o + delta_n)*w/c
%                  = (n_o + lambda/beat_length)*w/c
fiber.betas = [betas(1), betas(1) + 980e-9/beat_length *2*pi/sim.lambda0;...
               betas(2), betas(2) + 980e-9/beat_length /c;...
               betas(3), betas(3);... % dispersion are the same if delta_n is a constant
               betas(4), betas(4);...
               betas(5), betas(5)];

% Load default parameters like 
%
% loading fiber.betas and fiber.SR based on your multimode folder above
% sim.Raman_model = 1; Use isotropic Raman model
% sim.gain_model = 0; Don't use gain model = passive propagation
% sim.sw = true; Include shock term
% sim.gpu_yes = true; Use GPU (default to true)
% sim.step_method = 'RK4IP'; Use RK4IP
% sim.single_yes = true; Use single-precision
% sim.adaptive_deltaZ.model = 1; Use adaptive-step model
% ......
%
% Please check this function for details.
[fiber,sim] = load_default_GMMNLSE_propagate(fiber,sim,'single_mode'); % load default parameters for "fiber" and "sim"

%% Gain info
% Please find details of all the parameters in "gain_info.m" if not specified here.
% Note that the use of single spatial mode is different from multi-spatial modes.
% Activating "reuse_data" or "linear_oscillator_model" requires setting other parameters.
% Check the example or "gain_info.m".
gain_rate_eqn.cross_section_filename = 'Liekki Yb_AV_20160530.txt';
gain_rate_eqn.core_diameter = 6; % um
gain_rate_eqn.cladding_diameter = 125; % um
gain_rate_eqn.core_NA = 0.12;
gain_rate_eqn.absorption_wavelength_to_get_N_total = 920; % nm
gain_rate_eqn.absorption_to_get_N_total = 0.55; % dB/m
gain_rate_eqn.pump_wavelength = 976; % nm
gain_rate_eqn.copump_power = 4; % W
gain_rate_eqn.counterpump_power = 0; % W
gain_rate_eqn.reuse_data = false; % For a ring or linear cavity, the pulse will enter a steady state eventually.
                                  % If reusing the pump and ASE data from the previous roundtrip, the convergence can be much faster, especially for counterpumping.
gain_rate_eqn.linear_oscillator_model = 0; % For a linear oscillator, there are pulses from both directions simultaneously, which will deplete the gain;
                                           % therefore , the backward-propagating pulses need to be taken into account.
gain_rate_eqn.midx = 1; % the mode index; 1 represents the fundamental mode
gain_rate_eqn.mode_volume = 1; % the total number of available spatial modes in the fiber
gain_rate_eqn.t_rep = 25e-9; % Assume 40MHz here; s; the time required to finish a roundtrip (the inverse repetition rate of the pulse)
                             % This gain model solves the gain of the fiber under the steady-state condition; therefore, the repetition rate must be high compared to the lifetime of the doped ions.
gain_rate_eqn.tau = 840e-6; % lifetime of Yb in F_(5/2) state (Paschotta et al., "Lifetme quenching in Yb-doped fibers"); in "s"
gain_rate_eqn.export_N2 = false; % whether to export N2, the ion density in the upper state or not
gain_rate_eqn.ignore_ASE = true;
gain_rate_eqn.max_iterations = 50; % For counterpumping or considering ASE, iterations are required.
gain_rate_eqn.tol = 1e-5; % the tolerance for the above iterations
gain_rate_eqn.allow_coupling_to_zero_fields = false; % set this to false all the time for now
gain_rate_eqn.verbose = false; % show the information(final pulse energy) during iterations of computing the gain

%% Setup general parameters
Nt = 2^14; % the number of time points
time_window = 50; % ps
dt = time_window/Nt;
f = sim.f0+(-Nt/2:Nt/2-1)'/(Nt*dt); % THz
t = (-Nt/2:Nt/2-1)'*dt; % ps
c = 299792458; % m/s
lambda = c./(f*1e12)*1e9; % nm

% Precompute some parameters related to the gain to save the computational time
% Check "gain_info.m" for details.
[gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total] = gain_info( sim,gain_rate_eqn,ifftshift(lambda,1) );
gain_param = {gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total};

%% Initial condition
tfwhm = 0.5; % ps
total_energy = 0.1; % nJ
initial_pulse = build_MMgaussian(tfwhm, time_window, total_energy, 1, Nt);

% the pulse is mainly x-polarized perturbed by some tiny y-polarized components
random_polarization = rand(Nt,1)+1i*rand(Nt,1); random_polarization = random_polarization./abs(random_polarization);
initial_pulse.fields = [initial_pulse.fields initial_pulse.fields/10.*random_polarization]; % 20dB contrast

% Just to calculate the characteristic length here
[L_D_initial,L_NL_initial] = characteristic_lengths(abs(initial_pulse.fields(:,1)).^2,t,sim.f0,fiber.betas(3,:),1/fiber.SR);

%% Propagate
sim.ellipticity = 0; % linear polarization
prop_output_linear = GMMNLSE_propagate(fiber,initial_pulse,sim,gain_param);

%% Results
energy_linear = permute(trapz(abs(prop_output_linear.fields).^2,1)*dt/1e3,[3 2 1]);

[phi_linear,theta_linear] = calc_ellipticity( prop_output_linear.fields(:,:,end),0);

% total_field
total_field_linear = sum(abs(prop_output_linear.fields(:,:,end)).^2,2);

%% Plot Results
subplot(2,1,1);
h = plot(prop_output_linear.z,energy_linear(:,1));
set(h,'linewidth',2);
xlabel('Propagation distance (m)'); ylabel('Energy (nJ)');
title('x polarization');
subplot(2,1,2);
h = plot(prop_output_linear.z,energy_linear(:,2));
set(h,'linewidth',2);
xlabel('Propagation distance (m)'); ylabel('Energy (nJ)');
title('y polarization');