close all; clearvars;

outermost_folder = '../../../../';
MMTools_path = [outermost_folder 'MMTools/'];
addpath([MMTools_path 'GMMNLSE/']);

save_name = 'Mamyshev';
save_path = './';

%% Gain info
gain_rate_eqn.cross_section_filename = 'Liekki Yb_AV_20160530.txt';
gain_rate_eqn.core_diameter = 6; % um
gain_rate_eqn.cladding_diameter = 125; % um
gain_rate_eqn.core_NA = 0.12;
gain_rate_eqn.absorption_wavelength_to_get_N_total = 920; % nm
gain_rate_eqn.absorption_to_get_N_total = 0.55; % dB/m
gain_rate_eqn.pump_wavelength = 976; % nm
gain_rate_eqn.copump_power = 0; % W
gain_rate_eqn.counterpump_power = 0; % W
gain_rate_eqn.reuse_data = true;
gain_rate_eqn.linear_oscillator_model = 2;
gain_rate_eqn.midx = 1;
gain_rate_eqn.mode_volume = 1; % the total number of available modes in the fiber
gain_rate_eqn.tau = 840e-6; % lifetime of Yb in F_(5/2) state (Paschotta et al., "Lifetme quenching in Yb-doped fibers"); in "s"
gain_rate_eqn.export_N2 = true;
gain_rate_eqn.ignore_ASE = false;
gain_rate_eqn.max_iterations = 50;
gain_rate_eqn.tol = 1e-5;
gain_rate_eqn.allow_coupling_to_zero_fields = false;
gain_rate_eqn.verbose = true; % show the information(final pulse energy) during iterations

%% Setup fiber parameters
% General parameters
sim.lambda0 = 1030e-9;
sim.f0 = 2.99792458e-4/sim.lambda0;
sim.deltaZ = 2000e-6;
sim.mpa_yes = false;
sim.progress_bar = false;
sim.save_period = 0.1;
sim.gpu_yes = false;

num_modes = 1;

% -------------------------------------------------------------------------
% ------------------------------- Arm 1 (6um) -----------------------------
% -------------------------------------------------------------------------
sim6 = sim;
sim6.progress_bar_name = 'SMF (6um)';

[fiber6,sim6] = load_default_GMMNLSE_propagate([],sim6,'single_mode');

% -------------------------------------------------------------------------
% SMF
fiber_6a = fiber6;
fiber_6a.L0 = 3;

% Gain fiber
sim_Gain_6 = sim;
sim_Gain_6.gain_model = 4;
sim_Gain_6.progress_bar_name = 'Gain (6um)';
fiber_Gain_6.L0 = 3;
[fiber_Gain_6,sim_Gain_6] = load_default_GMMNLSE_propagate(fiber_Gain_6,sim_Gain_6,'single_mode');

% SMF
fiber_6b = fiber6;
fiber_6b.L0 = 0.3;

% -------------------------------------------------------------------------
% ----------------------------------- All ---------------------------------
% -------------------------------------------------------------------------
fiber_cavity = [fiber_6b fiber_Gain_6 fiber_6a ...   % arm 1
                fiber_6a fiber_Gain_6 fiber_6b];  % arm 2
sim_cavity = [sim6 sim_Gain_6 sim6 ...  % arm 1
              sim6 sim_Gain_6 sim6]; % arm 2

arm = {1:3 4:6};

%% Setup general cavity parameters
max_rt = 2000;
N = 2^14;
time_window = 100; % ps
dt = time_window/N;
f = sim.f0+(-N/2:N/2-1)'/(N*dt); % THz
t = (-N/2:N/2-1)*dt; % ps
c = 299792458; % m/s
lambda = c./(f*1e12)*1e9; % nm
OC = 0.8;
loss = 0.1;
tol_convergence = 1e-5;
force_stop = struct('tol_convergence',1e-4,'tol_fluctuation',5e-4,'min_rt',30);

%% Filter parameters
spectral_filter = struct('bw',4e-9, ...    % bandwidth
                         'cw',{1027e-9,1033e-9}); % central wavelength
filter_factor = @(center_lambda,bandwidth_lambda) exp(-(f-2.99792458e-4/center_lambda).^6/(2.99792458e-4/center_lambda^2*bandwidth_lambda/(2*sqrt(log(2))))^6);

%% Setup initial conditions
tfwhm = 1; % ps
total_energy = 10; % nJ
pedestal_energy = 0.01; % nJ

num_pulses = 3;
prop_output = build_noisy_MMgaussian(tfwhm, inf, time_window, total_energy,pedestal_energy,1,N,0.01,{'ifft',0},ones(1,num_pulses)*sqrt(1/num_pulses),-time_window/2+(1:num_pulses)*time_window/(num_pulses+1));
prop_output.fields = sum(prop_output.fields,2);
prop_output = gaussian_spectral_filter(prop_output, sim.f0, spectral_filter(2).cw, spectral_filter(2).bw,1,true); % Filter
prop_output.ASE_forward = zeros(N,1);
prop_output.ASE_backward = zeros(N,1);

%% Saved field information
L0 = sum([fiber_cavity.L0]);
save_num = int64(L0/sim.save_period + 1);
save_num = double(save_num);
saved_z = zeros(1,save_num);
field = cell(1,max_rt);
ASE.forward = cell(1,max_rt); ASE.backward = cell(1,max_rt);
ASE_forward_out = zeros(N,num_modes,max_rt);
N2 = cell(1,max_rt);
pump = cell(1,max_rt);
splice_z = cumsum([fiber_cavity.L0]);
filter_displacement = sim.save_period/25;
splice_z = [splice_z(1:3) splice_z(3)+filter_displacement splice_z(4:end)+filter_displacement];
output_field = zeros(N,num_modes,max_rt);
output_field2 = zeros(N,num_modes,max_rt);
max_save_per_fiber = 20;

%% Load gain parameters
L_air = 5; % 1 is the free-space length
c = 299792458; % m/s
v = 1/fiber_cavity(1).betas(2)*1e12; % velocity in the fiber
gain_rate_eqn.t_rep = L0/v + L_air/c; % s
gain_rate_eqn.deltaT = gain_rate_eqn.t_rep/2;

modulation_freq = 200e3; % Hz
pump_power = 3; % W

gain_rate_eqn_copumping = gain_rate_eqn;
gain_rate_eqn_counterpumping = gain_rate_eqn;
gain_rate_eqn_copumping.copump_power = pump_power; % W
gain_rate_eqn_counterpumping.counterpump_power = gain_rate_eqn_copumping.copump_power; % W

[gain_rate_eqn_copumping,cross_sections_pump_copumping,cross_sections_copumping,overlap_factor_copumping,N_total_copumping] = gain_info( sim_Gain_6,gain_rate_eqn_copumping,ifftshift(lambda,1) );
gain_param_copumping = {gain_rate_eqn_copumping,cross_sections_pump_copumping,cross_sections_copumping,overlap_factor_copumping,N_total_copumping};
[gain_rate_eqn_counterpumping,cross_sections_pump_counterpumping,cross_sections_counterpumping,overlap_factor_counterpumping,N_total_counterpumping] = gain_info( sim_Gain_6,gain_rate_eqn_counterpumping,ifftshift(lambda,1) );
gain_param_counterpumping = {gain_rate_eqn_counterpumping,cross_sections_pump_counterpumping,cross_sections_counterpumping,overlap_factor_counterpumping,N_total_counterpumping};

gain_param = {gain_param_counterpumping gain_param_copumping};

%% Run the cavity simulation
if sim_cavity(1).gpu_yes
    reset(gpuDevice);
end

func = analyze_sim;

% Initialize some parameters
output_energy = zeros(max_rt,1);
output_energy_ASE = zeros(max_rt,1);
time_delay = 0;
rt_num = 0;
pulse_survives = true;
rate_gain_saved_data = [];
time = 1/modulation_freq/4;
largeZ = true;
while rt_num < max_rt
    current_z = 0;
    zn = 1;
    rt_num = rt_num +1;
    
    t_iteration_start = tic;
    cprintf('*[1 0.5 0.31]','Iteration %d', rt_num);
    % -----------------------------------------------------------------
    for i = 1:2
        
        % Propagation inside fibers
        for j = arm{i}
            prop_output = GMMNLSE_propagate(fiber_cavity(j), prop_output, sim_cavity(j),gain_param{ceil(j/3)},rate_gain_saved_data);
            %time_delay = time_delay + prop_output.t_delay;
            
            % Save the information
            if ismember(j,[2,5])
                prop_output.fields = prop_output.fields.forward;
            end
            [saved_field,saved_z_this_fiber] = func.extract_saved_field(prop_output.fields,max_save_per_fiber,current_z,prop_output.z);
            field{rt_num}(:,:,zn:zn+size(saved_field,3)-1) = saved_field;
            saved_z(zn:zn+size(saved_field,3)-1) = saved_z_this_fiber;
            
            % Save the gain info
            if ismember(j,[2,5])
                rate_gain_saved_data = prop_output.saved_data;
                
                saved_N2 = func.extract_saved_field( prop_output.N2,max_save_per_fiber,current_z,sim.save_period );
                N2{rt_num}(:,:,zn:zn+size(saved_N2,3)-1) = saved_N2;
                saved_ASE_forward = func.extract_saved_field( prop_output.Power.ASE_forward,max_save_per_fiber,current_z,sim.save_period );
                ASE.forward{rt_num}(:,:,zn:zn+size(saved_ASE_forward,3)-1) = saved_ASE_forward;
                saved_ASE_backward = func.extract_saved_field( prop_output.Power.ASE_backward,max_save_per_fiber,current_z,sim.save_period );
                ASE.backward{rt_num}(:,:,zn:zn+size(saved_ASE_backward,3)-1) = saved_ASE_backward;
                switch i
                    case 1
                        saved_pump_backward = func.extract_saved_field( prop_output.Power.pump_backward,max_save_per_fiber,current_z,sim.save_period );
                        pump{rt_num}(:,:,zn:zn+size(saved_pump_backward,3)-1) = saved_pump_backward;
                    case 2
                        saved_pump_forward = func.extract_saved_field( prop_output.Power.pump_forward,max_save_per_fiber,current_z,sim.save_period );
                        pump{rt_num}(:,:,zn:zn+size(saved_pump_forward,3)-1) = saved_pump_forward;
                end
                idx = zn+size(saved_field,3)-1;
            elseif ismember(j,[3,6])
                N2{rt_num}(:,:,zn+1:zn+size(saved_field,3)-1) = 0;
                pump{rt_num}(:,:,zn+1:zn+size(saved_field,3)-1) = 0;
                ASE.forward{rt_num}(:,:,zn+1:zn+size(saved_field,3)-1) = zeros(size(saved_field)-[0,0,1]);
                ASE.backward{rt_num}(:,:,zn+1:zn+size(saved_field,3)-1) = zeros(size(saved_field)-[0,0,1]);
            else
                N2{rt_num}(:,:,zn:zn+size(saved_field,3)-1) = 0;
                pump{rt_num}(:,:,zn:zn+size(saved_field,3)-1) = 0;
                ASE.forward{rt_num}(:,:,zn:zn+size(saved_field,3)-1) = zeros(size(saved_field));
                ASE.backward{rt_num}(:,:,zn:zn+size(saved_field,3)-1) = zeros(size(saved_field));
                
                if ~exist('ASE_forward','var')
                    prop_output.ASE_forward = zeros(N,1);
                    prop_output.ASE_backward = zeros(N,1);
                else
                    prop_output.ASE_forward = ASE_forward;
                    prop_output.ASE_backward = ASE_backward;
                end
            end
            
            current_z = saved_z_this_fiber(end);
            zn = zn + size(saved_field,3)-1;
        end
        
        % Loss
        if i==1
            % Spectral filter (arm 1)
            if rt_num ~= 1
                close(fig_filter);
            end
            
            ASE_forward = filter_factor(spectral_filter(1).cw,spectral_filter(1).bw).*ASE.forward{rt_num}(:,:,idx); % Filter (ASE)
            ASE_forward = ASE_forward*(1-loss);
            ASE_backward = filter_factor(spectral_filter(2).cw,spectral_filter(2).bw).*ASE.backward{rt_num}(:,:,idx); % Filter (ASE)
            ASE_backward = (1-OC)*sqrt(1-loss)*ASE_backward;
            
            output_field2(:,:,rt_num) = prop_output.fields(:,:,end);
            [prop_output,fig_filter] = gaussian_spectral_filter(prop_output, sim.f0, spectral_filter(i).cw, spectral_filter(i).bw,3,true); % Filter
            
            prop_output.fields(:,:,end) = prop_output.fields(:,:,end)*sqrt(1-loss);
            
            % Save the field after the filter
            current_z = current_z + filter_displacement;
            zn = zn + 1;
        end
    end
    saved_z = saved_z(1:zn);
    
    % ASE
    ASE_forward_out(:,:,rt_num) = OC*ASE.forward{rt_num}(:,:,idx);
    ASE_forward = ASE.forward{rt_num}(:,:,idx)*(1-OC)*(1-loss);
    ASE_forward = filter_factor(spectral_filter(2).cw,spectral_filter(2).bw).*ASE_forward; % Filter (ASE)
    ASE_backward = filter_factor(spectral_filter(1).cw,spectral_filter(1).bw).*ASE.backward{rt_num}(:,:,idx); % Filter (ASE)
    ASE_backward = ASE_backward*(1-loss);
    
    % Output couplier
    output_field(:,:,rt_num) = sqrt(OC)*prop_output.fields(:,:,end);
    prop_output.fields = sqrt(1-OC)^2*sqrt(1-loss)*prop_output.fields(:,:,end);
    
    % -----------------------------------------------------------------
    % Spectral filter (after arm 2)
    close(fig_filter);
    [prop_output,fig_filter] = gaussian_spectral_filter(prop_output, sim.f0, spectral_filter(2).cw, spectral_filter(2).bw,3,true); % Filter
    
    % -----------------------------------------------------------------
    % Energy of the output field
    output_energy(rt_num) = sum(trapz(abs(output_field(:,:,rt_num)).^2))*prop_output.dt/10^3; % energy in nJ
    output_energy_ASE(rt_num) = trapz(ASE_forward_out(:,:,rt_num))/time_window*1e3; % mW

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
    % Update the repetition rate based on "time_delay"
    %gain_param{1}{1}.t_rep = gain_rate_eqn.t_rep + time_delay*1e-12;
    %gain_param{2}{1}.t_rep = gain_param{1}{1}.t_rep;
    
    % ---------------------------------------------------------------------
    % Characteristic lengths
    [ dispersion_length,nonlinear_length ] = characteristic_lengths( abs(output_field(:,1,rt_num)).^2,t,sim.f0,fiber_cavity(end).betas(3,:),1/fiber_cavity(end).SR );
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
end

%% Finish the simulation and save the data
% Clear reducdant parts of the data
N2 = N2(1:rt_num);
pump = pump(1:rt_num);
field = field(1:rt_num);
ASE.forward = ASE.forward(1:rt_num); ASE.backward = ASE.backward(1:rt_num);
ASE_forward_out = ASE_forward_out(:,:,1:rt_num);
output_field = output_field(:,:,1:rt_num);
output_field2 = output_field2(:,:,1:rt_num);
energy = output_energy(arrayfun(@any,output_energy)); % clear zero
energy_ASE = output_energy_ASE(1:rt_num);

% -------------------------------------------------------------------------
% Save the final output field
save([save_path save_name '.mat'], 't','f','output_field','output_field2','time_delay','energy','energy_ASE',...
                         'saved_z','splice_z','field',...
                         'N2','pump','ASE_forward_out','ASE','gain_param',...
                         'fiber_cavity','sim_cavity',... % cavity parameters
                         '-v7.3'); % saved mat file version
% -------------------------------------------------------------------------

close(fig,fig_filter,fig_evolution,fig_gain,fig_ASE);