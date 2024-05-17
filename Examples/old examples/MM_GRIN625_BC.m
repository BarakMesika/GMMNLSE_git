% This code runs the multimode Yb-doped fiber amplifier with the gain 
% rate equation.
%

clearvars; close all;

%% Add the folders of multimode files and others
addpath('../'); % add where many GMMNLSE-related functions like  "GMMNLSE_propagate" is
sim.cuda_dir_path = '../cuda'; % add cuda folder
fiber.MM_folder = '../Fibers/GIF625_10modes_CW1030nm/';
fiber.betas_filename = 'betas.mat';
fiber.S_tensors_filename = 'S_tensors_10modes.mat';

%% Gain info
gain_rate_eqn.MM_folder = fiber.MM_folder; % specify the folder with the eigenmode profiles
gain_rate_eqn.cross_section_filename = 'Liekki Yb_AV_20160530.txt';
gain_rate_eqn.saved_mat_filename = 'MM_YDFA_strong_waveguide_rate_eqn'; % the files used to temporarily store the data during iterations if necessary
gain_rate_eqn.reuse_data = false; % For a ring or linear cavity, the pulse will enter a steady state eventually.
                                  % If reusing the pump and ASE data from the previous roundtrip, the convergence can be much faster, especially for counterpumping.
gain_rate_eqn.linear_oscillator_model = 0; % For a linear oscillator, there are pulses from both directions simultaneously, which will deplete the gain;
                                           % therefore , the backward-propagating pulses need to be taken into account.
gain_rate_eqn.core_diameter = 62; % um
gain_rate_eqn.cladding_diameter = 125; % um
gain_rate_eqn.core_NA = 0.03; % in fact, this is only used in single-mode
gain_rate_eqn.absorption_wavelength_to_get_N_total = 915; % nm
gain_rate_eqn.absorption_to_get_N_total = 1.7; % dB/m
gain_rate_eqn.pump_wavelength = 976; % nm
gain_rate_eqn.copump_power = 0; % W
gain_rate_eqn.counterpump_power = 0; % W
gain_rate_eqn.midx = 1:10; % the mode index
gain_rate_eqn.mode_volume = 10; % the total number of available spatial modes in the fiber
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
time_window = 25; % ps
N = 2^14; % the number of time points
dt = time_window/N;
t = (-N/2:N/2-1)'*dt; % ps

fiber.L0 = 0.2; % m; the length of the gain fiber
save_num = 200;
sim.save_period = fiber.L0/save_num;
sim.single_yes = false;
sim.adaptive_deltaZ.model = 0; % turn adaptive-step off
sim.step_method = "RK4IP"; % use "MPA" instead of the default "RK4IP"
sim.deltaZ = 100e-6; % m; 
sim.lambda0 = 1030e-9; % central wavelength; in "m"
sim.Raman_model = 0;
%sim.progress_bar = false;
% sim.MPA.M = 5;

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

noise = randn(size(t))+1i*randn(size(t));
noise = noise/sqrt( dt*sum(abs(noise).^2)*1e-3 )*sqrt(1e-7);

% tmp = exp(-2.77*(t/300).^2);
% tmp = tmp/sqrt( dt*sum(abs(tmp).^2)*1e-3 )*sqrt(300);

tmp = exp(-2.77*(t/0.05).^2);
tmp = tmp/sqrt( dt*sum(abs(tmp).^2)*1e-3 )*sqrt(1);
% load('seed_GRIN625_GMNA.mat', 'p');
% tmp = p.'/sqrt( dt*sum(abs(p.').^2)*1e-3 )*sqrt(1);

E_all = 15;
E_LP01 = 0.10;
M_num = 3;
E_modes = rand(M_num,1);
E_modes = [E_modes; zeros(9-M_num,1)];
% E_modes = [1 1 1 1 0 0 0 0 0]';
E_modes = E_modes/sum( E_modes )*(1 - E_LP01)*E_all;
E_modes = [E_LP01*E_all; E_modes];
% plot(E_modes, 'bo')
% sum(E_modes(1))

for ii=1:gain_rate_eqn.mode_volume
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

%%
for ig = 20:10:60 
    E_all = 36;
    E_LP01 = ig/100;
    M_num = 4;

    dirName  = ['GIF625_BC\SeedModeNum_' num2str(M_num) '\Fundamental_' num2str(E_LP01*100) '_phaseRND9_\'];
    mkdir(dirName);
    data.t = t;
    data.lambda = lambda;
    data.distance = (0:save_num)*sim.save_period;
    data.f = f;
    data.f0 = sim.f0;
    data.seed = tmp;
    save([dirName 'simParam'], 'data');

    for ij=1:300
        E_modes = rand(M_num,1);
        E_modes = [E_modes; zeros(9-M_num,1)];
        E_modes = E_modes/sum( E_modes )*(1 - E_LP01)*E_all;
        E_modes = [E_LP01*E_all; E_modes];
        
        E_modes = [0.1 0.20 0.13 0.14 0.32 0.11 0 0 0 0]*E_all;
        if ij==1
            E_modes = E_modes + randn(1,10)*0;
        else
            E_tmp = randn(1,10)+1i*randn(1,10);
            E_tmp = E_tmp/sqrt(sum(abs(E_tmp).^2))*randn(1,1)/0.5;
            E_modes = E_modes + E_tmp;
        end
        
        for ii=1:gain_rate_eqn.mode_volume
%             input_field.fields(:,ii) = sqrt(E_modes(ii))*tmp + noise;
            input_field.fields(:,ii) = sqrt(E_modes(ii))*tmp + noise;
        end
        output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);
        uamp = single( output_field.fields );
        fName = ['data_' num2str(ij,'%03.0f')];
        save([dirName fName], 'uamp', 'E_modes');
    end
end

%% Plot results
cmap = linspecer(gain_rate_eqn.mode_volume);
% Energy
E = squeeze( sum(abs(output_field.fields).^2,1) )*dt/1e3;
distance = (0:save_num)*sim.save_period;
figure;
% yyaxis left
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
xlim([-5 5]);
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
xlim([800 1350]);
%% plot evolution
figure;
for jj=1:save_num
    for ii=1:gain_rate_eqn.mode_volume
        plot(fftshift(lambda),( abs(fftshift(ifft(output_field.fields(:,ii,jj)),1)).^2 ),...
            'DisplayName', ['Mode:' num2str(ii)], 'LineWidth', 2, 'Color', cmap(ii,:));
        hold on
    end
    hold off
    legend
    xlabel('Wavelength (nm)');
    ylabel('Intensity (a.u.)');
    title(['The spectrum of YDFA' '   z:' num2str(distance(jj)) '[m]']);
    xlim([800 1600]);
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
    xlim([-5 5]);
    xlabel('Time (ps)');
    ylabel('Intensity (W)');
    title(['The field of YDFA' '   z:' num2str(distance(jj)) '[m]']);
    drawnow
end