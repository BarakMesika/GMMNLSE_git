% This code runs the multimode Yb-doped fiber amplifier with the gain 
% rate equation.
%

clearvars; close all;

%% Add the folders of multimode files and others
addpath('../..'); % add where many GMMNLSE-related functions like  "GMMNLSE_propagate" is
sim.cuda_dir_path = '../../cuda'; % add cuda folder
fiber.MM_folder = '../../Fibers/PLMA-YDF-25-250-VIII/';
fiber.betas_filename = 'betas.mat';
fiber.S_tensors_filename = 'S_tensors_6modes.mat';

%% Gain info
gain_rate_eqn.MM_folder = fiber.MM_folder; % specify the folder with the eigenmode profiles
gain_rate_eqn.cross_section_filename = 'Liekki Yb_AV_20160530.txt';
gain_rate_eqn.saved_mat_filename = 'MM_YDFA_strong_waveguide_rate_eqn'; % the files used to temporarily store the data during iterations if necessary
gain_rate_eqn.reuse_data = false; % For a ring or linear cavity, the pulse will enter a steady state eventually.
                                  % If reusing the pump and ASE data from the previous roundtrip, the convergence can be much faster, especially for counterpumping.
gain_rate_eqn.linear_oscillator_model = 0; % For a linear oscillator, there are pulses from both directions simultaneously, which will deplete the gain;
                                           % therefore , the backward-propagating pulses need to be taken into account.
gain_rate_eqn.core_diameter = 25; % um
gain_rate_eqn.cladding_diameter = 250; % um
gain_rate_eqn.core_NA = 0.06; % in fact, this is only used in single-mode
gain_rate_eqn.absorption_wavelength_to_get_N_total = 915; % nm
gain_rate_eqn.absorption_to_get_N_total = 1.7; % dB/m
gain_rate_eqn.pump_wavelength = 976; % nm
gain_rate_eqn.copump_power = 15; % W
gain_rate_eqn.counterpump_power = 0; % W
gain_rate_eqn.midx = 1:6; % the mode index
gain_rate_eqn.mode_volume = 6; % the total number of available spatial modes in the fiber
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

fiber.L0 = 3.5; % m; the length of the gain fiber
save_num = 350;
sim.save_period = fiber.L0/save_num;
sim.single_yes = false;
sim.adaptive_deltaZ.model = 0; % turn adaptive-step off
sim.step_method = 'MPA'; % use "MPA" instead of the default "RK4IP"
sim.deltaZ = 500e-6; % m; 
sim.lambda0 = 1030e-9; % central wavelength; in "m"
sim.Raman_model = 0;
%sim.progress_bar = false;

% Rate-equation gain
sim.gain_model = 2;
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
input_field.fields = zeros([size(t,1) 6]);

tmp = exp(-2.77*(t/0.6).^2);
tmp = tmp/sqrt( dt*sum(abs(tmp).^2)*1e-3 )*sqrt(1);

input_field.fields(:,1) = tmp;
input_field.fields(:,2) = tmp;
input_field.fields(:,3) = tmp;
input_field.fields(:,4) = tmp;
input_field.fields(:,5) = tmp;
input_field.fields(:,6) = tmp;
% plot(fftshift(lambda), abs(fftshift(ifft(tmp))).^2)
% xlim([1000 1100]);grid on;
% plot(t, abs(tmp).^2)
% input_field = build_MMgaussian(0.3, time_window, 3, length(gain_rate_eqn.midx), N);
% input_field.fields(:,2:end)=0;
%% Propagation
if sim.gpu_yes
    reset(gpuDevice);
end
%%
[Mfield, dx] = get_modes_fields('D:\GMMNLSE\Fibers\PLMA-YDF-25-250-VIII\', '1030');
filter = exp(-2.77*((lambda-1030)/12).^2);
NPEsat = 15000;
NPEmod = 0.98;
uRT{1} = input_field.fields;
cmap = lines(size(input_field.fields,2));
% saveDir = 'D:\GMMNLSE\SimData\Oscillator_PLMA-YDF-25-250-VIII\PH012um\';
R = (2:2:26)*1e-6;
fiber.gain_coeff = 2*log(1000)/fiber.L0*500;
fiber.gain_fwhm = 40e-9;
fiber.saturation_intensity  = 50e-9/(pi*(25e-6/2)^2);
%%
for jj=1:13
    tmp = exp(-2.77*(t/0.6).^2);
    tmp = tmp/sqrt( dt*sum(abs(tmp).^2)*1e-3 )*sqrt(1);
    input_field.fields = zeros([size(t,1) 6]);
    input_field.fields(:,1) = tmp;
    input_field.fields(:,2) = tmp;
    input_field.fields(:,3) = tmp;
    input_field.fields(:,4) = tmp;
    input_field.fields(:,5) = tmp;
    input_field.fields(:,6) = tmp;
    uRT{1} = input_field.fields;
    for RT=1:100
        %Spectral filter
        spec = ifft(input_field.fields,[],1);
        tmp = bsxfun(@times, filter, spec); 
        input_field.fields = fft(tmp,[],1);
    
        %Pinhole
        input_field.fields = get_modes_pinhole(Mfield, input_field.fields, dx, R(jj), [5e-6 5e-6]);
    
        %Gain fiber
        output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);
    
        %Saturable absorption
        SA = NPEmod./(1 + ( abs(output_field.fields(:,:,end)).^2 )/NPEsat );
        uRT{RT+1} = sqrt(SA).*output_field.fields(:,:,end);
        input_field.fields = sqrt(1-SA).*output_field.fields(:,:,end);
    
        close(gcf);
        figure(RT);set(gcf, 'Position', [100 100 1000 600])
        subplot(1,2,1)
        for ii=1:size(input_field.fields,2)
            plot(t,abs( uRT{RT+1}(:,ii) ).^2, 'Color', cmap(ii,:));
            hold on
    %         plot(t,abs( input_field.fields(:,ii) ).^2, 'LineStyle',...
    %             ':', 'Color', cmap(ii,:), 'LineWidth', 2);
            xlabel('Time [ps]');ylabel('Power [W]');
        
        end
        xlabel('Time [ps]');ylabel('Power [W]');
        drawnow
    
        subplot(1,2,2)
        for ii=1:size(input_field.fields,2)
            plot(lambda,abs( (ifft(uRT{RT+1}(:,ii))) ).^2, 'Color', cmap(ii,:));
            hold on
    %         plot(lambda,abs( (ifft(input_field.fields(:,ii))) ).^2, 'LineStyle',...
    %             ':', 'Color', cmap(ii,:), 'LineWidth', 2);
            xlabel('Time [ps]');ylabel('Power [W]');
        
        end
        xlabel('\lambda [nm]');ylabel('Spectrum [a.u.]');
        xlim([980 1120])
        drawnow
    
        E(RT) = permute(sum(trapz(abs(uRT{RT+1}).^2),2)*dt/1e3,[3 2 1]);
        fprintf('E: %d\n', E(RT))
    
        fname = ['data_RT_' num2str(RT,'%.3d')];
        saveas(gcf, fullfile(saveDir(jj,:), fname), 'tiffn');
        saveas(gcf, fullfile(saveDir(jj,:), fname), 'fig');
        uout = uRT{RT+1};
        save(fullfile(saveDir(jj,:), fname), 'uout', 'input_field');
        pause(0.2)
        if E(RT)<1e-2
            break
        end
    end
end
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
for ii=1:200
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
%%
for ii=1:save_num
    plot(t, abs(output_field.fields(:,:,ii)).^2,...
                'DisplayName', ['Mode' num2str(ii)])
    drawnow;
end