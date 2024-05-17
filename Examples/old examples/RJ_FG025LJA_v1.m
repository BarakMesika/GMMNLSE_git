% This code runs the multimode Yb-doped fiber amplifier with the gain 
% rate equation.
%

clearvars; close all;

%% Add the folders of multimode files and others
addpath('..'); % add where many GMMNLSE-related functions like  "GMMNLSE_propagate" is
sim.cuda_dir_path = '../cuda'; % add cuda folder
fiber.MM_folder = '../Fibers/FG025LJA_20/';
fiber.betas_filename = 'betas.mat';
fiber.S_tensors_filename = 'S_tensors_17modes.mat';

%% Gain info
gain_rate_eqn.MM_folder = fiber.MM_folder; % specify the folder with the eigenmode profiles
gain_rate_eqn.cross_section_filename = 'Liekki Yb_AV_20160530.txt';
gain_rate_eqn.saved_mat_filename = 'MM_YDFA_strong_waveguide_rate_eqn'; % the files used to temporarily store the data during iterations if necessary
gain_rate_eqn.reuse_data = false; % For a ring or linear cavity, the pulse will enter a steady state eventually.
                                  % If reusing the pump and ASE data from the previous roundtrip, the convergence can be much faster, especially for counterpumping.
gain_rate_eqn.linear_oscillator_model = 0; % For a linear oscillator, there are pulses from both directions simultaneously, which will deplete the gain;
                                           % therefore , the backward-propagating pulses need to be taken into account.
gain_rate_eqn.core_diameter = 25; % um
gain_rate_eqn.cladding_diameter = 125; % um
gain_rate_eqn.core_NA = 0.1; % in fact, this is only used in single-mode
gain_rate_eqn.absorption_wavelength_to_get_N_total = 915; % nm
gain_rate_eqn.absorption_to_get_N_total = 7.5; % dB/m
gain_rate_eqn.pump_wavelength = 976; % nm
gain_rate_eqn.copump_power = 0; % W
gain_rate_eqn.counterpump_power = 0; % W
gain_rate_eqn.midx = 1:17; % the mode index
gain_rate_eqn.mode_volume = 17; % the total number of available spatial modes in the fiber
gain_rate_eqn.downsampling_factor = 1; % downsample the eigenmode profiles to run faster
gain_rate_eqn.t_rep = 1/1e6; % assume 1MHz here; s; the time required to finish a roundtrip (the inverse repetition rate of the pulse)
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
N = 2^15; % the number of time points
dt = time_window/N;
t = (-N/2:N/2-1)'*dt; % ps

fiber.L0 = 0.2; % m; the length of the gain fiber
save_num = 20;
sim.save_period = fiber.L0/save_num;
sim.single_yes = false;
sim.adaptive_deltaZ.model = 0; % turn adaptive-step off
sim.step_method = 'RK4IP'; % use "MPA" instead of the default "RK4IP"
sim.deltaZ = 200e-6; % m; 
sim.lambda0 = 1035e-9; % central wavelength; in "m"
sim.Raman_model = 0;
%sim.progress_bar = false;

% Rate-equation gain
sim.gain_model = 0;
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
input_field.fields = zeros([size(t,1) gain_rate_eqn.mode_volume]);

tmp = sqrt( exp(-2.77*(t/0.6).^2) );
GDD = 12;
tmp = fft( ifft(tmp).*exp(1i*GDD*(f-sim.f0).^2) );
tmp = tmp/sqrt( dt*sum(abs(tmp).^2)*1e-3 )*sqrt(1500);
plot(t, abs(tmp).^2)

noise = randn(size(tmp))+1i*randn(size(tmp));
noise = noise/sqrt( dt*sum(abs(noise).^2)*1e-3 )*sqrt(1e-5);

E = zeros(gain_rate_eqn.mode_volume,1);
E(1:6) = 0.2;
E(7:end) = 0.001;
for ii=1:gain_rate_eqn.mode_volume
    input_field.fields(:,ii) = sqrt(E(ii))*tmp+noise;
end


% plot(fftshift(lambda), abs(fftshift(ifft( input_field.fields(:,1) ))).^2)
% xlim([1000 1100]);grid on;
% plot(t, abs( input_field.fields(:,:) ).^2)
% input_field = build_MMgaussian(0.3, time_window, 3, length(gain_rate_eqn.midx), N);
% input_field.fields(:,2:end)=0;
%% Propagation
if sim.gpu_yes
    reset(gpuDevice);
end
%%
% fiber.gain_coeff = 8*log(1000)/fiber.L0;
% fiber.gain_fwhm = 40e-9;
% fiber.saturation_intensity  = 1e-6;

%%
for ii = 21:100
    ii
    E = zeros(gain_rate_eqn.mode_volume,1);
    Etmp = rand(6,1);
    Etmp = Etmp/sum(Etmp);
    E(1:6) = Etmp;
    E(7:end) = 0.001;
    for ij=1:gain_rate_eqn.mode_volume
        input_field.fields(:,ij) = sqrt(E(ij))*tmp+noise;
    end
    output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);
    uout = output_field.fields;
    uin = input_field.fields;
    z = output_field.z;
    t = t;
    fname = ['H:\GMMNLSE\Examples\RJ_FG025LJA_Data\data' num2str(ii) '.mat'];
    save(fname, 'uout', 'uin', 'z', 't', 'f', 'lambda');
end
%% Plot results
cmap = jet(gain_rate_eqn.mode_volume);
% Energy
E = squeeze( sum(abs(output_field.fields).^2,1) )*dt/1e3;
distance = (0:save_num)*sim.save_period;
figure;
yyaxis left
for ii=1:gain_rate_eqn.mode_volume
    plot(distance, E(ii,:),'DisplayName', ['Mode:' num2str(ii)], 'LineWidth', 2,...
        'Color', cmap(ii,:));
    hold on
end
hold off
% yyaxis right
% plot(distance, sum(E,1),'DisplayName', ['Mode: all'], 'LineWidth', 2, 'Color', 'r');
legend
xlabel('Propagation length (m)');
ylabel('Energy (nJ)');
title('Energy');


%% Field

figure;
for ii=1:gain_rate_eqn.mode_volume
    plot(t, abs(output_field.fields(:,ii,end)).^2,'DisplayName', ['Mode:' num2str(ii)], 'LineWidth', 2,...
        'Color', cmap(ii,:));
    hold on
end
hold off
legend
% xlim([-20 20]);
xlabel('Time (ps)');
ylabel('Intensity (W)');
title('The final output field of YDFA');

%% Spectrum
figure;
for ii=1:gain_rate_eqn.mode_volume
    plot(fftshift(lambda),( abs(fftshift(ifft(output_field.fields(:,ii,end)),1)).^2 ),...
        'DisplayName', ['Mode:' num2str(ii)], 'LineWidth', 2, 'Color', cmap(ii,:));
    hold on
end
hold off
legend
xlabel('Wavelength (nm)');
ylabel('Intensity (a.u.)');
title('The final output spectrum of YDFA');
xlim([600 1650]);
%% plot evolution
figure;
for jj=1:save_num
    for ii=1:gain_rate_eqn.mode_volume
        plot(fftshift(lambda),10*log10( abs(fftshift(ifft(output_field.fields(:,ii,jj)),1)).^2 ),...
            'DisplayName', ['Mode:' num2str(ii)], 'LineWidth', 2, 'Color', cmap(ii,:));
        hold on
    end
    hold off
    legend
    xlabel('Wavelength (nm)');
    ylabel('Intensity (a.u.)');
    title(['The spectrum of YDFA' '   z:' num2str(distance(jj)) '[m]']);
    xlim([500 1650]);
%     xlim([-50 50]);
    drawnow
end
%%
figure;
for jj=1:save_num
    for ii=1:gain_rate_eqn.mode_volume
        plot(t, abs(output_field.fields(:,ii,jj)).^2,'DisplayName', ['Mode:' num2str(ii)], 'LineWidth', 2,...
            'Color', cmap(ii,:));
        hold on
    end
    hold off
    legend
    xlim([-20 20]);
    xlabel('Time (ps)');
    ylabel('Intensity (W)');
    title(['The field of YDFA' '   z:' num2str(distance(jj)) '[m]']);
    drawnow
end