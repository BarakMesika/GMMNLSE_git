% This code computes the linear Mamyshev oscillator with only one gain
% fiber without any passive one.
% Note that the final pump power and the inversion, N2, are the same at
% each point of the gain fiber for co- and counter-pumping because the gain
% can't respond to the pulse in time and sees only the average effect
% including both the forward and backward propagating pulses.
close all; clearvars;

addpath('../../');

save_name = 'linear_oscillator';
save_path = './';

%% Gain info
% Please find details of all the parameters in "gain_info.m" if not specified here.
% Note that the use of single spatial mode is different from multi-spatial modes.
% "reuse_data" and "linear_oscillator_model" are activated and some related parameters are set.
gain_rate_eqn.cross_section_filename = 'Liekki Yb_AV_20160530.txt';
gain_rate_eqn.core_diameter = 6; % um
gain_rate_eqn.cladding_diameter = 125; % um
gain_rate_eqn.core_NA = 0.12;
gain_rate_eqn.absorption_wavelength_to_get_N_total = 920; % nm
gain_rate_eqn.absorption_to_get_N_total = 0.55; % dB/m
gain_rate_eqn.pump_wavelength = 976; % nm
gain_rate_eqn.copump_power = 0; % W; it's set below
gain_rate_eqn.counterpump_power = 0; % W; it's set below
gain_rate_eqn.reuse_data = true; % For a ring or linear cavity, the pulse will enter a steady state eventually.
                                 % If reusing the pump and ASE data from the previous roundtrip, the convergence can be much faster, especially for counterpumping.
gain_rate_eqn.linear_oscillator_model = 2; % For a linear oscillator, there are pulses from both directions simultaneously, which will deplete the gain;
                                           % therefore , the backward-propagating pulses need to be taken into account.
gain_rate_eqn.save_all_in_RAM = true; % Save all the required info in the RAM instead of MAT files with the filename defined below
gain_rate_eqn.saved_mat_filename = 'linear_oscillator_rate_eqn'; % if "save_all_in_RAM" is false, this is the name of the saved MAT files; if it's true, you can put any string since it's unused.
gain_rate_eqn.midx = 1; % the mode index; 1 represents the fundamental mode
gain_rate_eqn.mode_volume = 1; % the total number of available spatial modes in the fiber
gain_rate_eqn.tau = 840e-6; % lifetime of Yb in F_(5/2) state (Paschotta et al., "Lifetme quenching in Yb-doped fibers"); in "s"
gain_rate_eqn.export_N2 = true; % whether to export N2, the ion density in the upper state or not
gain_rate_eqn.ignore_ASE = false;
gain_rate_eqn.max_iterations = 50; % For counterpumping or considering ASE, iterations are required.
gain_rate_eqn.tol = 1e-5; % the tolerance for the above iterations
gain_rate_eqn.allow_coupling_to_zero_fields = false; % set this to false all the time for now
gain_rate_eqn.verbose = true; % show the information(final pulse energy) during iterations of computing the gain

%% Setup fiber parameters
% General parameters
sim.lambda0 = 1030e-9; % m
sim.f0 = 2.99792458e-4/sim.lambda0; % THz
sim.deltaZ = 1000e-6; % um
sim.progress_bar = false;
sim.save_period = 0.05;
sim.gpu_yes = false; % For only the fundamental mode, running with CPU is faster if the number of points is lower than 2^(~18).
sim.adaptive_deltaZ.model = 0; % Adaptive-step method doesn't support reuse_data, linear_oscillator_model, and save_all_in_RAM.

% -------------------------------------------------------------------------
% -------------------------------- Arm (6um) ------------------------------
% -------------------------------------------------------------------------
% Gain fiber
sim_Gain = sim;
sim_Gain.gain_model = 4;
fiber_Gain.L0 = 3; % the length of the gain fiber

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
[fiber_Gain,sim_Gain] = load_default_GMMNLSE_propagate(fiber_Gain,sim_Gain,'single_mode');

% -------------------------------------------------------------------------
% ----------------------------------- All ---------------------------------
% -------------------------------------------------------------------------
fiber_cavity = [fiber_Gain fiber_Gain];
sim_cavity = [sim_Gain sim_Gain];

arm = {1 2};

%% Setup general cavity parameters
max_rt = 100; % maximum number of roundtrips
N = 2^14; % the number of points
time_window = 100; % ps
dt = time_window/N;
f = sim.f0+(-N/2:N/2-1)'/(N*dt); % THz
t = (-N/2:N/2-1)*dt; % ps
c = 299792458; % m/s
lambda = c./(f*1e12)*1e9; % 
OC = 0.8; % output coupler
loss= 0.5;
num_modes = 1; % only fundamental mode is considered in this example
tol_convergence = 1e-5; % the tolerance of pulse energy convergence for the oscillator
force_stop = struct('tol_convergence',1e-4,'tol_fluctuation',5e-4,'min_rt',30); % If the roundtrip number is larger than min_rt, it'll start to check the trend of convergence.
                                                                                % If it's still fluctuating within a certain tolerance, keep running; if fluctuating too much, the iteration will be forced to stop, showing that no stable solution is found with this oscillator parameters.

%% Filter parameters
spectral_filter = struct('bw',4e-9, ... % bandwidth
                         'cw',{1025e-9,1045e-9}); % central wavelength

% Similar to "spectral_filter" for the pulse; the spectral_filter for the ASE.
filter_factor = @(center_lambda,bandwidth_lambda) exp(-(f-2.99792458e-4/center_lambda).^2/(2.99792458e-4/center_lambda^2*bandwidth_lambda/(2*sqrt(log(2))))^2);

%% Setup initial conditions
tfwhm = 1; % ps
total_energy = 10; % nJ
pedestal_energy = 0.1; % nJ
prop_output = build_noisy_MMgaussian(tfwhm, inf, time_window, total_energy,pedestal_energy,num_modes,N,0.01);
prop_output.ASE_forward = zeros(N,1);
prop_output.ASE_backward = zeros(N,1);
            
%% Saved field information
field = cell(1,max_rt);
ASE.forward = cell(1,max_rt); ASE.backward = cell(1,max_rt);
N2 = cell(1,max_rt);
pump = cell(1,max_rt);
splice_z = cumsum([fiber_cavity.L0]);
filter_displacement = sim.save_period/25;
splice_z = [splice_z(1) splice_z(1)+filter_displacement splice_z(2)+filter_displacement];
output_field = zeros(N,num_modes,max_rt);
max_save_per_fiber = 100;

%% Reset the GPU
if sim_Gain.gpu_yes
    reset(gpuDevice);
end

%% Load gain parameters
L_air = 1; % 1 is the free-space length
c = 299792458; % m/s
v = 1/fiber_Gain.betas(2)*1e12; % velocity in the fiber
gain_rate_eqn.t_rep = sum([fiber_cavity.L0])/v + L_air/c; % s

gain_rate_eqn_copumping = gain_rate_eqn;
gain_rate_eqn_counterpumping = gain_rate_eqn;
gain_rate_eqn_copumping.copump_power = 3; % W
gain_rate_eqn_counterpumping.counterpump_power = gain_rate_eqn_copumping.copump_power; % W

[gain_rate_eqn_copumping,cross_sections_pump_copumping,cross_sections_copumping,overlap_factor_copumping,N_total_copumping] = gain_info( sim_Gain,gain_rate_eqn_copumping,ifftshift(lambda,1) );
gain_param_copumping = {gain_rate_eqn_copumping,cross_sections_pump_copumping,cross_sections_copumping,overlap_factor_copumping,N_total_copumping};
[gain_rate_eqn_counterpumping,cross_sections_pump_counterpumping,cross_sections_counterpumping,overlap_factor_counterpumping,N_total_counterpumping] = gain_info( sim_Gain,gain_rate_eqn_counterpumping,ifftshift(lambda,1) );
gain_param_counterpumping = {gain_rate_eqn_counterpumping,cross_sections_pump_counterpumping,cross_sections_counterpumping,overlap_factor_counterpumping,N_total_counterpumping};

gain_param = {gain_param_counterpumping gain_param_copumping};

%% Run the cavity simulation

func = analyze_sim;

% Initialize some parameters
output_energy = zeros(max_rt,1);
rt_num = 0;
pulse_survives = true;
rate_gain_saved_data = [];
while rt_num < max_rt
    time_delay = 0;
    current_z = 0;
    zn = 1;
    rt_num = rt_num +1;
    
    t_iteration_start = tic;
    cprintf('*[1 0.5 0.31]','Iteration %d', rt_num);
    % -----------------------------------------------------------------
    for i = 1:2
        % Propagation inside fibers
        for j = arm{i}
            prop_output = GMMNLSE_propagate(fiber_cavity(j), prop_output, sim_cavity(j),gain_param{j},rate_gain_saved_data);
            
            prop_output.fields = prop_output.fields.forward;
            rate_gain_saved_data = prop_output.saved_data;
            time_delay = time_delay + prop_output.t_delay;
            
            % Save the information
            [saved_field,saved_z_this_fiber] = func.extract_saved_field(prop_output.fields,max_save_per_fiber,current_z,prop_output.z);
            field{rt_num}(:,:,zn:zn+size(saved_field,3)-1) = saved_field;
            saved_z(zn:zn+size(saved_field,3)-1) = saved_z_this_fiber;
            % Save the gain info
            saved_N2 = func.extract_saved_field( prop_output.N2,max_save_per_fiber,current_z,sim.save_period );
            N2{rt_num}(:,:,zn:zn+size(saved_field,3)-1) = saved_N2;
            saved_ASE_forward = func.extract_saved_field( prop_output.Power.ASE_forward,max_save_per_fiber,current_z,sim.save_period );
            ASE.forward{rt_num}(:,:,zn:zn+size(saved_field,3)-1) = saved_ASE_forward;
            saved_ASE_backward = func.extract_saved_field( prop_output.Power.ASE_backward,max_save_per_fiber,current_z,sim.save_period );
            ASE.backward{rt_num}(:,:,zn:zn+size(saved_field,3)-1) = saved_ASE_backward;
            switch i
                case 2
                    saved_pump_forward = func.extract_saved_field( prop_output.Power.pump_forward,max_save_per_fiber,current_z,sim.save_period );
                    pump{rt_num}(:,:,zn:zn+size(saved_field,3)-1) = saved_pump_forward;
                case 1
                    saved_pump_backward = func.extract_saved_field( prop_output.Power.pump_backward,max_save_per_fiber,current_z,sim.save_period );
                    pump{rt_num}(:,:,zn:zn+size(saved_field,3)-1) = saved_pump_backward;
            end
            
            current_z = saved_z_this_fiber(end);
            zn = zn + size(saved_field,3)-1;
        end
        
        % Loss and spectral filter
        if i==1
            % Spectral filter (arm 1)
            if rt_num ~= 1
                close(fig_filter);
            end
            
            ASE_forward = filter_factor(spectral_filter(1).cw,spectral_filter(1).bw).*prop_output.Power.ASE_forward(:,:,end); % Filter (ASE)
            ASE_forward = ASE_forward*(1-loss);
            ASE_backward = filter_factor(spectral_filter(2).cw,spectral_filter(2).bw).*prop_output.Power.ASE_backward(:,:,end); % Filter (ASE)
            ASE_backward = (1-OC)*sqrt(1-loss)*ASE_backward;
            
            [prop_output,fig_filter] = gaussian_spectral_filter(prop_output, sim.f0, spectral_filter(1).cw, spectral_filter(1).bw,1,true); % Filter
            
            prop_output.fields(:,:,end) = prop_output.fields(:,:,end)*sqrt(1-loss);
            
            prop_output.ASE_forward = ASE_forward;
            prop_output.ASE_backward = ASE_backward;
            
            % Save the field after the filter
            current_z = current_z + filter_displacement;
            zn = zn + 1;
        end
    end
    saved_z = saved_z(1:zn);
    
    % ASE
    ASE_forward = prop_output.Power.ASE_forward(:,:,end)*(1-OC)*(1-loss);
    ASE_forward = filter_factor(spectral_filter(2).cw,spectral_filter(2).bw).*ASE_forward; % Filter (ASE)
    ASE_backward = filter_factor(spectral_filter(1).cw,spectral_filter(1).bw).*prop_output.Power.ASE_backward; % Filter (ASE)
    ASE_backward = ASE_backward*(1-loss);
    
    % ---------------------------------------------------------------------
    % Output coupler
    output_field(:,:,rt_num) = sqrt(OC)*prop_output.fields(:,:,end);
    prop_output.fields = sqrt(1-OC)*sqrt(1-loss)*prop_output.fields(:,:,end);
    
    % ---------------------------------------------------------------------
    % Spectral filter
    close(fig_filter);
    [prop_output,fig_filter] = gaussian_spectral_filter(prop_output, sim.f0, spectral_filter(2).cw, spectral_filter(2).bw,1,true); % Filter
    
    % ---------------------------------------------------------------------
    % Energy of the output field
    output_energy(rt_num) = sum(trapz(abs(output_field(:,:,rt_num)).^2))*prop_output.dt/1e3; % energy in nJ

    % If the energies stop changing, then we're done!
    if rt_num ~= 1
        close(fig);
    end
    warning('off')
    output_field(:,:,rt_num) = pulse_tracker(output_field(:,:,rt_num));
    warning('on');
    [converged_yes,force_stop_yes,fig] = check_convergence( output_energy,output_field(:,:,rt_num),f,t,tol_convergence,true,force_stop );
    
    % ---------------------------------------------------------------------
    % Display running time
    t_iteration_end = toc(t_iteration_start);
    t_iteration_spent = datevec(t_iteration_end/3600/24);
    fprintf(': %1u:%2u:%3.1f\n',t_iteration_spent(4),t_iteration_spent(5),t_iteration_spent(6));
    
    % ---------------------------------------------------------------------
    % ASE
    prop_output.ASE_forward = ASE_forward;
    prop_output.ASE_backward = ASE_backward;
    
    % Update the repetition rate based on "time_delay"
    gain_param{1}{1}.t_rep = gain_rate_eqn.t_rep + time_delay*1e-12;
    gain_param{2}{1}.t_rep = gain_param{1}{1}.t_rep;
    
    % ---------------------------------------------------------------------
    % Characteristic lengths
    [ dispersion_length,nonlinear_length ] = characteristic_lengths( abs(output_field(:,1,rt_num)).^2,t,sim.f0,fiber_Gain.betas(3,:),1/fiber_Gain.SR );
    fprintf('  L_DL=%4f(m)\n',dispersion_length);
    fprintf('  L_NL=%4f(m)\n',nonlinear_length);
    
    % ---------------------------------------------------------------------
    % Plot
    if rt_num ~= 1
        close(fig_evolution);
    end
    fig_evolution = func.analyze_fields_within_cavity(t,f,field{rt_num},saved_z,splice_z);
    
    if rt_num ~= 1
        close(fig_gain);
    end
    fig_gain = func.analyze_gain_within_cavity(saved_z,splice_z,pump{rt_num},N2{rt_num});
    
    if rt_num ~= 1
        close(fig_ASE);
    end
    ASE_plot.forward = ASE.forward{rt_num}; ASE_plot.backward = ASE.backward{rt_num};
    fig_ASE = func.analyze_ASE_within_cavity(f,ASE_plot,saved_z,splice_z);
    
    % ---------------------------------------------------------------------
    % Break if converged
    if converged_yes || force_stop_yes.convergence || force_stop_yes.fluctuation
        cprintf('blue','The field has converged!\n');
        break;
    end
    % Break if pulse dies
    if output_energy(rt_num) < 0.01
        disp('The pulse dies.');
        pulse_survives = false;
        break;
    end
end

%% Finish the simulation and save the data
% Clear reducdant parts of the data
field = field(1:rt_num);
ASE.forward = ASE.forward(1:rt_num); ASE.backward = ASE.backward(1:rt_num);
N2 = N2(1:rt_num);
pump = pump(1:rt_num);
output_field = output_field(:,:,1:rt_num);
energy = output_energy(arrayfun(@any,output_energy)); % clear zero

% -------------------------------------------------------------------------
% Save the final output field
if pulse_survives
    save([save_path save_name '.mat'], 't','f','output_field','time_delay','energy',...
                             'saved_z','splice_z','field',...
                             'pump','N2','ASE','gain_param',...
                             'fiber_cavity','sim_cavity',... % cavity parameters
                             '-v7.3'); % saved mat file version
end
% -------------------------------------------------------------------------

close(fig,fig_filter,fig_evolution,fig_gain,fig_ASE);