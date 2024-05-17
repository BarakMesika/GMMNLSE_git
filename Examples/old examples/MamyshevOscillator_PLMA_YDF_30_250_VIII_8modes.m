% This code runs the multimode Yb-doped fiber amplifier with the gain 
% rate equation.
%

clearvars; close all;

%% Add the folders of multimode files and others
addpath('../'); % add where many GMMNLSE-related functions like  "GMMNLSE_propagate" is
sim.cuda_dir_path = '../cuda'; % add cuda folder
fiber.MM_folder = '../Fibers/PLMA-YDF-30-250-VIII/';
fiber.betas_filename = 'betas.mat';
fiber.S_tensors_filename = 'S_tensors_8modes.mat';

%% Gain info
gain_rate_eqn.MM_folder = fiber.MM_folder; % specify the folder with the eigenmode profiles
gain_rate_eqn.cross_section_filename = 'Liekki Yb_AV_20160530.txt';
gain_rate_eqn.saved_mat_filename = 'MM_YDFA_strong_waveguide_rate_eqn'; % the files used to temporarily store the data during iterations if necessary
gain_rate_eqn.reuse_data = false; % For a ring or linear cavity, the pulse will enter a steady state eventually.
                                  % If reusing the pump and ASE data from the previous roundtrip, the convergence can be much faster, especially for counterpumping.
gain_rate_eqn.linear_oscillator_model = 0; % For a linear oscillator, there are pulses from both directions simultaneously, which will deplete the gain;
                                           % therefore , the backward-propagating pulses need to be taken into account.
gain_rate_eqn.core_diameter = 30; % um
gain_rate_eqn.cladding_diameter = 250; % um
gain_rate_eqn.core_NA = 0.06; % in fact, this is only used in single-mode
gain_rate_eqn.absorption_wavelength_to_get_N_total = 915; % nm
gain_rate_eqn.absorption_to_get_N_total = 0.9; % dB/m
gain_rate_eqn.pump_wavelength = 976; % nm
gain_rate_eqn.copump_power = 5; % W
gain_rate_eqn.counterpump_power = 0; % W
gain_rate_eqn.midx = 1:8; % the mode index
gain_rate_eqn.mode_volume = 8; % the total number of available spatial modes in the fiber
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
N = 2^11; % the number of time points
dt = time_window/N;
t = (-N/2:N/2-1)'*dt; % ps

fiber.L0 = 2.0; % m; the length of the gain fiber
save_num = 20;
sim.midx = 1:8;
sim.mode_volume = 8;
sim.save_period = fiber.L0/save_num;
sim.single_yes = true;
sim.adaptive_deltaZ.model = 0; % turn adaptive-step off
sim.step_method = "RK4IP"; % use "MPA" instead of the default "RK4IP"
sim.deltaZ = 500e-6; % m; 
sim.lambda0 = 1030e-9; % central wavelength; in "m"
sim.Raman_model = 0;
%sim.progress_bar = false;
% sim.MPA.M = 5;

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
input_field.fields = zeros([size(t,1) sim.mode_volume]);

noise = randn(size(t))+1i*randn(size(t));
noise = noise/sqrt( dt*sum(abs(noise).^2)*1e-3 )*sqrt(1e-7);

% tmp = exp(-2.77*(t/300).^2);
% tmp = tmp/sqrt( dt*sum(abs(tmp).^2)*1e-3 )*sqrt(300);

tmp = exp(-2.77*(t/0.15).^2);
tmp = tmp/sqrt( dt*sum(abs(tmp).^2)*1e-3 )*sqrt(1);
% load('seed_GRIN625_GMNA.mat', 'p');
% tmp = p.'/sqrt( dt*sum(abs(p.').^2)*1e-3 )*sqrt(1);

E_all = 0.1;
E_LP01 = 0.125;
M_num = 3;
% E_modes = rand(M_num,1);
% E_modes = [E_modes; zeros(9-M_num,1)];
E_modes = [1 1 1 1 1 1 1]';
E_modes = E_modes/sum( E_modes )*(1 - E_LP01)*E_all;
E_modes = [E_LP01*E_all; E_modes];
% plot(E_modes, 'bo')
% sum(E_modes(1))

for ii=1:sim.mode_volume
    input_field.fields(:,ii) = sqrt(E_modes(ii))*tmp + noise;
    
end


% plot(fftshift(lambda), abs(fftshift(ifft(tmp))).^2)
% xlim([1540 1560]);grid on;
% plot(t, abs(input_field.fields).^2)
% input_field = build_MMgaussian(tfwhm, time_window, total_energy, length(gain_rate_eqn.midx), N);
% input_field.fields(:,2:end)=0;
%% Propagation
if sim.gpu_yes
    reset(gpuDevice);
end
% gain_param = []; % simulation wothout gain

output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);


FilterR = exp(-2.77*((lambda - 1033)/4).^4);
FilterB = exp(-2.77*((lambda - 1027)/4).^4);
plot(lambda, FilterB, lambda, FilterR)
RTNum = 500;
SaveDir = 'F:\GMMNLSE\Examples\MamyshevOscillator_PLMA_YDF_30_250_VIII_8modes_Data\';
save([SaveDir 'OscData'], 'FilterR', 'FilterB', 't', 'lambda', 'f');
for rt=1:RTNum
    for ii=1:sim.mode_volume
        input_field.fields(:,ii) = fft( FilterR.*ifft(output_field.fields(:,ii,end)) )+ 0*noise;
    end
    AllFields = single( input_field.fields );
    
    fiber.L0 = 1.0; % m; the length of the gain fiber
    save_num = 10;
    sim.save_period = fiber.L0/save_num;
    sim.gain_model = 0;
    output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);
    AllFields = cat(3, AllFields, single( output_field.fields ));
    
    fiber.L0 = 2.0; % m; the length of the gain fiber
    save_num = 20;
    sim.save_period = fiber.L0/save_num;
    sim.gain_model = 4;
    input_field.fields = squeeze( output_field.fields(:,:,end) );
    output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);
    AllFields = cat(3, AllFields, single( output_field.fields ));
    
    fiber.L0 = 1.0; % m; the length of the gain fiber
    save_num = 10;
    sim.save_period = fiber.L0/save_num;
    sim.gain_model = 0;
    input_field.fields = squeeze( output_field.fields(:,:,end) );
    output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);
    AllFields = cat(3, AllFields, single( output_field.fields ));
    
    for ii=1:sim.mode_volume
        input_field.fields(:,ii) = fft( FilterB.*ifft(output_field.fields(:,ii,end)) )+ 0*noise;
    end
    AllFields = cat(3, AllFields, single( input_field.fields ));
    
    fiber.L0 = 1.0; % m; the length of the gain fiber
    save_num = 10;
    sim.save_period = fiber.L0/save_num;
    sim.gain_model = 0;
    output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);
    AllFields = cat(3, AllFields, single( output_field.fields ));
    
    fiber.L0 = 2.0; % m; the length of the gain fiber
    save_num = 20;
    sim.save_period = fiber.L0/save_num;
    sim.gain_model = 4;
    input_field.fields = squeeze( output_field.fields(:,:,end) );
    output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);
    AllFields = cat(3, AllFields, single( output_field.fields ));
    
    fiber.L0 = 1.0; % m; the length of the gain fiber
    save_num = 10;
    sim.save_period = fiber.L0/save_num;
    sim.gain_model = 0;
    input_field.fields = squeeze( output_field.fields(:,:,end) );
    output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);
    AllFields = cat(3, AllFields, single( output_field.fields ));
    
    save([SaveDir 'RTdata' num2str(rt,'%03.f')], 'AllFields');
    rt = rt+1
end
%% Plot results
cmap = linspecer(sim.mode_volume);
% Energy
E = squeeze( sum(abs(AllFields).^2,1) )*dt/1e3;
distance = (0:size(AllFields,3)-1)*sim.save_period;
figure;
% yyaxis left
for ii=1:sim.mode_volume
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
grid on

%% Field
figure;
for ii=1:sim.mode_volume
    plot(t, abs(output_field.fields(:,ii,end)).^2,'DisplayName', ['Mode:' num2str(ii)], 'LineWidth', 2,...
        'Color', cmap(ii,:));
    hold on
end
hold off
legend
% xlim([-5 5]);
xlabel('Time (ps)');
ylabel('Intensity (W)');
title('The final output field');

%% Spectrum
figure;
for ii=1:sim.mode_volume
    plot(fftshift(lambda),( abs(fftshift(ifft(output_field.fields(:,ii,end)),1)).^2 ),...
        'DisplayName', ['Mode:' num2str(ii)], 'LineWidth', 2, 'Color', cmap(ii,:));
    hold on
end
hold off
legend
xlabel('Wavelength (nm)');
ylabel('Intensity (a.u.)');
title('The final output spectrum');
xlim([900 1150]);
%% plot evolution
figure;
for jj=1:size(AllFields,3)
    for ii=1:sim.mode_volume
        plot(fftshift(lambda),( abs(fftshift(ifft(AllFields(:,ii,jj)),1)).^2 ),...
            'DisplayName', ['Mode:' num2str(ii)], 'LineWidth', 2, 'Color', cmap(ii,:));
        hold on
    end
    hold off
    legend
    xlabel('Wavelength (nm)');
    ylabel('Intensity (a.u.)');
    title(['The spectrum' '   z:' num2str(distance(jj)) '[m]']);
    xlim([900 1150]);
%     xlim([-50 50]);
    drawnow
end
%%
figure;
for jj=1:size(AllFields,3)
    for ii=1:sim.mode_volume
        plot(t, abs(AllFields(:,ii,jj)).^2,'DisplayName', ['Mode:' num2str(ii)], 'LineWidth', 2,...
            'Color', cmap(ii,:));
        hold on
    end
    hold off
    legend
%     xlim([-5 5]);
    xlabel('Time (ps)');
    ylabel('Intensity (W)');
    title(['The field' '   z:' num2str(distance(jj)) '[m]']);
    drawnow
end