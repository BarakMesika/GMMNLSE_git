% This code runs the multimode Yb-doped fiber amplifier with the gain 
% rate equation.
%

clearvars; close all;

%% Add the folders of multimode files and others
addpath('../..'); % add where many GMMNLSE-related functions like  "GMMNLSE_propagate" is
sim.cuda_dir_path = '../cuda'; % add cuda folder
fiber.MM_folder = '../Fibers/GRIN625_10modes/';
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
gain_rate_eqn.core_diameter = 64; % um
gain_rate_eqn.cladding_diameter = 125; % um
gain_rate_eqn.core_NA = 0.06; % in fact, this is only used in single-mode
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
time_window = 30; % ps
N = 2^14; % the number of time points
dt = time_window/N;
t = (-N/2:N/2-1)'*dt; % ps

fiber.L0 = 3; % m; the length of the gain fiber
save_num = 10;
sim.save_period = fiber.L0/save_num;
sim.single_yes = false;
sim.adaptive_deltaZ.model = 0; % turn adaptive-step off
sim.step_method = 'RK4IP'; % use "MPA" instead of the default "RK4IP"
sim.deltaZ = 2000e-6; % m; 
sim.lambda0 = 1030e-9; % central wavelength; in "m"
sim.Raman_model = 0;
%sim.progress_bar = false;
sim.MPA.M = 10;

% Rate-equation gain
sim.gain_model = 2;
fiber.gain_coeff = 45*log(10)/(10*fiber.L0);
fiber.gain_fwhm = 45e-9;
fiber.saturation_intensity  = 150e-9/(pi*(25e-6/2)^2);
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

tmp = exp(-2.77*(t/0.2).^2);
% tmp = fft( ifft(tmp).*exp(+1i*0*(f-sim.f0).^2) );
tmp = tmp/sqrt( dt*sum(abs(tmp).^2)*1e-3 )*sqrt(10);
% tmp = load('seed_GRIN625_MO.mat');
Menergy = exp(-2.77*(((1:gain_rate_eqn.mode_volume)-0)/150000).^2);
for ii=1:gain_rate_eqn.mode_volume
    input_field.fields(:,ii) = 1*tmp + 0*noise;
    
end

% input_field.fields(:,1) = tmp + noise;
% input_field.fields(:,2) = tmp + noise;
% input_field.fields(:,3) = tmp + noise;
% input_field.fields(:,4) = tmp + noise;
% input_field.fields(:,5) = tmp + noise;
% input_field.fields(:,6) = tmp + noise;
% input_field.fields(:,7) = tmp + noise;
% input_field.fields(:,8) = tmp + noise;
% input_field.fields(:,9) = tmp + noise;
% input_field.fields(:,10) = tmp + noise;

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
dF = 2;
filter = exp(-2.77*((fftshift(f)-sim.f0-dF)/0.4).^2);
filterR = repmat(filter,1,gain_rate_eqn.mode_volume);
filter = exp(-2.77*((fftshift(f)-sim.f0+dF)/0.4).^2);
filterB = repmat(filter,1,gain_rate_eqn.mode_volume);
% plot(lambda, fftshift(filterR), lambda, fftshift(filterB))

output_field.fields = input_field.fields;
E(1,:) = squeeze( dt*sum(abs(output_field.fields(:,:,end)).^2,1)*1e-3 );
uRT{1} = input_field.fields;
for jj=1:50
    spec = fftshift( ifft(output_field.fields,[],1), 1);
    input_field.fields = fft(fftshift( spec.*filterR.*exp(+1i*0*(f-sim.f0-dF).^2) ,1 ),[],1);
    output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);
    uRT_R{jj} = output_field.fields; 
    
    spec = fftshift( ifft(output_field.fields,[],1), 1);
    input_field.fields = fft(fftshift( spec.*filterB.*exp(+1i*0*(f-sim.f0+dF).^2) ,1 ),[],1);
    output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);
    uRT_B{jj} = output_field.fields;
    
    
    plot(fftshift(lambda), abs( fftshift( ifft(output_field.fields(:,:,end),[],1), 1) ).^2, 'LineWidth', 1)
    xlim([850 1250])
    legend;
    drawnow
    E(jj+1,:) = squeeze( dt*sum(abs(output_field.fields(:,:,end)).^2,1)*1e-3 );
    
    fprintf('E_t_o_t: %i DE:%i \n', sum(E(jj+1,:),2), abs(sum(E(jj+1,:),2)-sum(E(jj,:),2)))
    if abs(sum(E(jj+1,:),2)-sum(E(jj,:),2))<1e-3 || sum(E(jj+1,:),2)<0.1
        break;
    end
end
%% Plot results
E = squeeze( dt*sum(abs(output_field.fields).^2,1)*1e-3 );

spec = fftshift( ifft(output_field.fields,[],1), 1);
filter = exp(-2.77*((fftshift(f)-sim.f0)/3.7).^8);
filter2 = repmat(filter,1,10,save_num+1);
specF = spec.*filter2;
uampF = fft(fftshift( specF ,1 ),[],1);
EF = squeeze( dt*sum(abs(uampF).^2,1)*1e-3 );
%%
% v = VideoWriter('BC_V4_chirped10_noRaman.avi');
% open(v);
figure('pos', [50 150 1450 350])
for ii=1:numel(uRT_B)
    for ij = 1:(save_num+1)
        subplot(1,2,1)
        plot(fftshift(lambda), ( abs( fftshift( ifft( uRT_B{ii}(:,:,ij) ,[],1), 1) ).^2 ))
        xlim([ 1030-150 1030+150]);
        ylim([0 35])
        xlabel('\lambda [nm]')
        ylabel('Spectrum [a.u.]')
    
        subplot(1,2,2)
        plot(t, abs( uRT_B{ii}(:,:,ij) ).^2 )
        xlim([-15 15])
        ylim([0 3e5])
        xlabel('Time [ps]')
        ylabel('Power [W]')
        drawnow
    end
    
    for ij = 1:(save_num+1)
        subplot(1,2,1)
        plot(fftshift(lambda), ( abs( fftshift( ifft( uRT_R{ii}(:,:,ij) ,[],1), 1) ).^2 ))
        xlim([ 1030-150 1030+150]);
        ylim([0 35])
        xlabel('\lambda [nm]')
        ylabel('Spectrum [a.u.]')

        subplot(1,2,2)
        plot(t, abs( uRT_R{ii}(:,:,ij) ).^2 )
        xlim([-15 15])
        ylim([0 3e5])
        xlabel('Time [ps]')
        ylabel('Power [W]')    
    
        drawnow
%     frame = getframe(gcf);
%     writeVideo(v,frame);
    end
end
% close(v);
