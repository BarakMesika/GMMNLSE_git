function varargout = GMMNLSE_rategain( sim,gain_rate_eqn,...
                                            cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,...
                                            z_points_all,save_points,num_large_steps_persave,...
                                            initial_condition,...
                                            prefactor, SRa_info, SRb_info, SK_info, omegas, D_op, haw, hbw,...
                                            saved_data)
%GMMNLSE_RATEGAIN It attains the field after propagation inside the gain 
%medium solved by the rate equations.
%   
% The computation of this code is based on
%   1. Lindberg et al., "Accurate modeling of high-repetition rate ultrashort pulse amplification in optical fibers", Scientific Reports (2016)
%   2. Chen et al., "Optimization of femtosecond Yb-doped fiber amplifiers for high-quality pulse compression", Opt. Experss (2012)
%   3. Gong et al., "Numerical modeling of transverse mode competition in strongly pumped multimode fiber lasers and amplifiers", Opt. Express (2007)
%
%   Please go to "gain_info.m" file to find the information about some input arguments.
%   The info of some other input arguments are inside "GMMNLSE_propagate.m"
%
%   Brief introduction:
%       First calculate the amplification(for SM) or the gain term in
%       GMMNLSE(for MM) from rate equations and send this into MPA or split
%       step algorithm for the pulse propagation.
%       If there's backward-propagating fields, that is, ASE or
%       counterpumping, iterations are necessary to get a more accurate
%       result.

N = size(initial_condition.fields,1);
num_modes = size(initial_condition.fields,2);

%% Pump direction
if gain_rate_eqn.copump_power == 0
    if gain_rate_eqn.counterpump_power == 0
        gain_rate_eqn.pump_direction = 'co'; % use 'co' for zero pump power
    else
        gain_rate_eqn.pump_direction = 'counter';
    end
else
    if gain_rate_eqn.counterpump_power == 0
        gain_rate_eqn.pump_direction = 'co';
    else
        gain_rate_eqn.pump_direction = 'bi';
    end
end

doesnt_need_iterations = isequal(gain_rate_eqn.pump_direction,'co') && gain_rate_eqn.ignore_ASE ; % There's no need to compute backward propagation if copumping and ignoring ASE, so the first forward propagation solves everything.

%% To consider the iterations between forward and backward propagations, "all the information" about signal_fields and Power are saved to "mat" files to save the memory.
% These files will be consistently updated and loaded throughout iterations.
% For more details about the iteration process, please refer to 
% "Lindberg et al., "Accurate modeling of high-repetition rate ultrashort pulse amplification in optical fibers", Scientific Reports (2016)"

if doesnt_need_iterations && ~gain_rate_eqn.reuse_data
    num_segments = 1;
    segments = save_points;
else
    % "single" precision is assumed here.
    %    single: 4 bytes
    %    double: 8 bytes
    if sim.single_yes
        precision = 4;
    else
        precision = 8;
    end
    mem_complex_number = precision*2;
    % The size of the variables:
    variable_size.signal_fields = 2*N*num_modes*z_points_all; % forward and backward
    variable_size.Power_pump = 2*z_points_all; % forward and backward
    variable_size.Power_ASE  = 2*N*num_modes*z_points_all; % forward and backward
    variable_size.signal_out_in_solve_gain_rate_eqn = N*num_modes^2*sim.MPA.M; % Because it sometimes blows up the GPU memory here, so I added it.
    variable_size.cross_sections = 2*N;
    variable_size.overlap_factor = numel(overlap_factor);
    variable_size.N_total = numel(N_total);
    variable_size.FmFnN = numel(FmFnN);
    variable_size.GammaN = numel(GammaN);
    var_field = fieldnames(variable_size);
    used_memory = 0;
    for i = 1:length(var_field)
        used_memory = used_memory + variable_size.(var_field{i});
    end
    used_memory = used_memory*mem_complex_number;

    if gain_rate_eqn.save_all_in_RAM && ~sim.gpu_yes
        num_segments = 1;
    else
        num_segments = ceil(used_memory/gain_rate_eqn.memory_limit);
    end
    if num_segments == 1
        segments = z_points_all;
    else
        num_each_segment = ceil(z_points_all/num_segments);
        segments = [num_each_segment*ones(1,num_segments-1) z_points_all-num_each_segment*(num_segments-1)];
        if segments(end) == 0
            segments = segments(1:end-1);
        end
    end
end

%% Initialization of "Power" and "signal_fields"
% They're both in the frequency domain.
if isequal(gain_rate_eqn.pump_direction,'co')
    segment_idx = 1;
else % 'counter', 'bi'
    segment_idx = num_segments;
end
if gain_rate_eqn.save_all_in_RAM && sim.gpu_yes
    z_points = z_points_all;
else
    z_points = segments(segment_idx);
end
[signal_fields,signal_fields_backward,Power_pump_forward,Power_ASE_forward,Power_pump_backward,Power_ASE_backward] = initialization('both',sim,gain_rate_eqn,segment_idx,num_segments,N,num_modes,z_points,initial_condition);
if gain_rate_eqn.save_all_in_RAM && sim.gpu_yes
    [signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward] = mygpuArray2(1,segments(1),signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward);
end

%% Propagations
% If counter/bi-pumping, backward-propagate first without the signal to set up the inversion level.
load_saved_files = false;
not_first_run = false;
if gain_rate_eqn.reuse_data
    if gain_rate_eqn.save_all_in_RAM
        not_first_run = ~isempty(saved_data);
        if not_first_run
            Power_pump_forward     = saved_data.Power_pump_forward;
            Power_pump_backward    = saved_data.Power_pump_backward;
            Power_ASE_forward      = saved_data.Power_ASE_forward;
            Power_ASE_backward     = saved_data.Power_ASE_backward;
            signal_fields          = saved_data.signal_fields;
            signal_fields_backward = saved_data.signal_fields_backward;
            
            if sim.gpu_yes
                [signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward] = mygpuArray2(1,segments(1),signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward);
            end
            
            load_saved_files = true;
        end
    else
        current_caller_path = cd;
        not_first_run  = (exist(fullfile(current_caller_path, sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,segment_idx)), 'file') == 2);
        if not_first_run
            load(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,1),'Power_pump_forward','Power_pump_backward','Power_ASE_forward','Power_ASE_backward','signal_fields','signal_fields_backward');
            [signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward] = mygpuArray(signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward);
            
            load_saved_files = true;
        end
    end
end
if any(strcmp(gain_rate_eqn.pump_direction,{'counter','bi'}))
    if ~not_first_run
        [signal_fields_backward,        Power_pump_backward,Power_ASE_backward]                                     = gain_propagate('backward',sim,gain_rate_eqn,z_points_all,segments,load_saved_files,save_points,num_large_steps_persave,N,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,initial_condition,signal_fields,signal_fields_backward,Power_pump_forward, Power_ASE_forward );
    end

    load_saved_files = true;
end

% Start the pulse propagation.
if isequal(sim.step_method,'MPA') % MPA
    if gain_rate_eqn.export_N2
        [signal_fields_backward,t_delay,Power_pump_forward,Power_ASE_forward,signal_fields,N2,full_iterations_hist] = gain_propagate('forward', sim,gain_rate_eqn,z_points_all,segments,load_saved_files,save_points,num_large_steps_persave,N,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,initial_condition,signal_fields,signal_fields_backward,Power_pump_backward,Power_ASE_backward);
    else
        [signal_fields_backward,t_delay,Power_pump_forward,Power_ASE_forward,signal_fields,   full_iterations_hist] = gain_propagate('forward', sim,gain_rate_eqn,z_points_all,segments,load_saved_files,save_points,num_large_steps_persave,N,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,initial_condition,signal_fields,signal_fields_backward,Power_pump_backward,Power_ASE_backward);
    end
else % split-step, RK4IP
    if gain_rate_eqn.export_N2
        [signal_fields_backward,t_delay,Power_pump_forward,Power_ASE_forward,signal_fields,N2]                      = gain_propagate('forward', sim,gain_rate_eqn,z_points_all,segments,load_saved_files,save_points,num_large_steps_persave,N,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,initial_condition,signal_fields,signal_fields_backward,Power_pump_backward,Power_ASE_backward);
    else
        [signal_fields_backward,t_delay,Power_pump_forward,Power_ASE_forward,signal_fields]                         = gain_propagate('forward', sim,gain_rate_eqn,z_points_all,segments,load_saved_files,save_points,num_large_steps_persave,N,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,initial_condition,signal_fields,signal_fields_backward,Power_pump_backward,Power_ASE_backward);
    end
end

if sim.gpu_yes
    if sim.single_yes
        energy = zeros(1,gain_rate_eqn.max_iterations,'single','gpuArray');
    else
        energy = zeros(1,gain_rate_eqn.max_iterations,'gpuArray');
    end
else
    if sim.single_yes
        energy = zeros(1,gain_rate_eqn.max_iterations,'single');
    else
        energy = zeros(1,gain_rate_eqn.max_iterations);
    end
end
if ~gain_rate_eqn.ignore_ASE
    energy_ASE_forward  = zeros(1,gain_rate_eqn.max_iterations);
    energy_ASE_backward = zeros(1,gain_rate_eqn.max_iterations);
    if sim.single_yes
        energy_ASE_forward  = single(energy_ASE_forward);
        energy_ASE_backward = single(energy_ASE_backward);
    end
    if sim.gpu_yes
        energy_ASE_forward  = gpuArray(energy_ASE_forward);
        energy_ASE_backward = gpuArray(energy_ASE_backward);
    end
end
% Start the iterations
if ~doesnt_need_iterations
    load_saved_files = true;

    energy(1) = calc_total_energy(signal_fields{end},N,initial_condition.dt); % nJ
    if ~gain_rate_eqn.ignore_ASE
        energy_ASE_forward(1)  = sum(Power_ASE_forward{end}) /(N*initial_condition.dt); % W
        energy_ASE_backward(1) = sum(Power_ASE_backward{end})/(N*initial_condition.dt);
    end
    if gain_rate_eqn.verbose
        fprintf('Gain rate equation, iteration %u: pulse energy = %7.6g(nJ)\n',1,energy(1));
    end
    break_yes = false;
    for i = 2:gain_rate_eqn.max_iterations
        % Backward propagation
                [signal_fields_backward,        Power_pump_backward,Power_ASE_backward]                                     = gain_propagate('backward',sim,gain_rate_eqn,z_points_all,segments,load_saved_files,save_points,num_large_steps_persave,N,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,initial_condition,signal_fields,signal_fields_backward,Power_pump_forward, Power_ASE_forward );

        % Forward propagation
        if isequal(sim.step_method,'MPA') % MPA
            if gain_rate_eqn.export_N2
                [signal_fields_backward,t_delay,Power_pump_forward,Power_ASE_forward,signal_fields,N2,full_iterations_hist] = gain_propagate('forward', sim,gain_rate_eqn,z_points_all,segments,load_saved_files,save_points,num_large_steps_persave,N,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,initial_condition,signal_fields,signal_fields_backward,Power_pump_backward,Power_ASE_backward);
            else
                [signal_fields_backward,t_delay,Power_pump_forward,Power_ASE_forward,signal_fields,   full_iterations_hist] = gain_propagate('forward', sim,gain_rate_eqn,z_points_all,segments,load_saved_files,save_points,num_large_steps_persave,N,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,initial_condition,signal_fields,signal_fields_backward,Power_pump_backward,Power_ASE_backward);
            end
        else % split-stepp, RK4IP
            if gain_rate_eqn.export_N2
                [signal_fields_backward,t_delay,Power_pump_forward,Power_ASE_forward,signal_fields,N2]                      = gain_propagate('forward', sim,gain_rate_eqn,z_points_all,segments,load_saved_files,save_points,num_large_steps_persave,N,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,initial_condition,signal_fields,signal_fields_backward,Power_pump_backward,Power_ASE_backward);
            else
                [signal_fields_backward,t_delay,Power_pump_forward,Power_ASE_forward,signal_fields]                         = gain_propagate('forward', sim,gain_rate_eqn,z_points_all,segments,load_saved_files,save_points,num_large_steps_persave,N,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,initial_condition,signal_fields,signal_fields_backward,Power_pump_backward,Power_ASE_backward);
            end
        end

        % Check convergence
        energy(i) = calc_total_energy(signal_fields{end},N,initial_condition.dt); % nJ
        if ~gain_rate_eqn.ignore_ASE
            energy_ASE_forward(i)  = sum(Power_ASE_forward{end})/(N*initial_condition.dt); %W
            energy_ASE_backward(i) = sum(Power_ASE_backward{1}) /(N*initial_condition.dt); %W
        end
        if gain_rate_eqn.verbose
            fprintf('Gain rate equation, iteration %u: pulse energy = %7.6g(nJ)\n',i,energy(i));
            if ~gain_rate_eqn.ignore_ASE
                fprintf('                                 forward  ASE power = %7.6g(mW)\n',energy_ASE_forward(i) *1e3);
                fprintf('                                 backward ASE power = %7.6g(mW)\n',energy_ASE_backward(i)*1e3);
            end
        end
        if energy(i-1)==0 || abs((energy(i)-energy(i-1))./energy(i-1)) < gain_rate_eqn.tol
            if ~gain_rate_eqn.ignore_ASE
                ASE_converged = ( abs((energy_ASE_forward(i)-energy_ASE_forward(i-1))./energy_ASE_forward(i-1)) < gain_rate_eqn.tol ) && ...
                                ( abs((energy_ASE_backward(i)-energy_ASE_backward(i-1))./energy_ASE_backward(i-1)) < gain_rate_eqn.tol );
            else % ignore ASE
                ASE_converged = true;
            end

            if ASE_converged
                break_yes = true;
            end
        end
        % Plot the convergence
        if gain_rate_eqn.verbose && i > 10 && ~(i==10+1 && break_yes)
            if i == 10+1
                fig_gain_iterations = figure('Name','Gain iterations');
            end
            figure(fig_gain_iterations);
            h = plot(1:i,energy(1:i));
            xlabel('Iterations'); ylabel('Pulse energy (nJ)');
            title('Convergence');
            set(h,'linewidth',2); set(gca,'fontsize',14);
            drawnow;
            if break_yes
                close(fig_gain_iterations);
            end
        end
        if break_yes
            break;
        end
        if i == gain_rate_eqn.max_iterations
            if ~doesnt_need_iterations || gain_rate_eqn.reuse_data
                clean_saved_mat(gain_rate_eqn,num_segments);
            end

            error('GMMNLSE_gain_rate_eqn:NotConvergedError',...
                'The iteration of forward and backward propagation of the gain fiber doesn''t converge to a steady state within %u iterations.',gain_rate_eqn.max_iterations);
        end
    end
end

% Output:
% Power_out is in frequency domain while field_out is transformed into time domain.
saved_z_points = 1:num_large_steps_persave:z_points_all;
if num_segments == 1 || (gain_rate_eqn.save_all_in_RAM && sim.gpu_yes)
    % Transform the current "signal_fields" and "Power..." into arrays.
    if sim.gpu_yes
        [signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward] = mygather(signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward);
        [signal_fields_out,signal_fields_backward_out,Power_pump_forward_out,Power_pump_backward_out,Power_ASE_forward_out,Power_ASE_backward_out] = deal(signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward);
    else
        [signal_fields_out,signal_fields_backward_out,Power_pump_forward_out,Power_pump_backward_out,Power_ASE_forward_out,Power_ASE_backward_out] = deal(signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward);
    end
    signal_fields_out          = cell2mat(signal_fields_out);
    signal_fields_backward_out = cell2mat(signal_fields_backward_out);
    % Change the size back to (N,num_modes).
    Power_ASE_forward_out  = cellfun(@(P) permute(P,[5 3 1 2 4]), Power_ASE_forward_out, 'UniformOutput',false);
    Power_ASE_backward_out = cellfun(@(P) permute(P,[5 3 1 2 4]), Power_ASE_backward_out,'UniformOutput',false);

    % Transform them into arrays
    Power_pump_forward_out  = cell2mat(Power_pump_forward_out);
    Power_pump_backward_out = cell2mat(Power_pump_backward_out);
    Power_ASE_forward_out   = fftshift(cell2mat(Power_ASE_forward_out ),1);
    Power_ASE_backward_out  = fftshift(cell2mat(Power_ASE_backward_out),1);
    
    if doesnt_need_iterations && ~gain_rate_eqn.reuse_data
        if ~not_first_run
            saved_z_points = 1:size(signal_fields_backward_out,3);
        end
        Power_out = struct('pump_forward',Power_pump_forward_out,'pump_backward',0,...
                           'ASE_forward',                      0,'ASE_backward', 0);
        signal_fields_out = struct('forward', fft(signal_fields_out),...
                                   'backward',fft(signal_fields_backward_out(:,:,saved_z_points)));
    else
        Power_out = struct('pump_forward',Power_pump_forward_out(:,:,saved_z_points),'pump_backward',Power_pump_backward_out(:,:,saved_z_points),...
                           'ASE_forward', Power_ASE_forward_out (:,:,saved_z_points),'ASE_backward' , Power_ASE_backward_out(:,:,saved_z_points));
        signal_fields_out = struct('forward', fft(signal_fields_out(:,:,saved_z_points)),...
                                   'backward',fft(signal_fields_backward_out(:,:,saved_z_points)));
    end
    
    % Reverse the order and save the data for the linear oscillator scheme
    % for the next round
    if gain_rate_eqn.reuse_data
        if gain_rate_eqn.linear_oscillator
            reverse_direction = @(x) flip(x,3);
            Power_pump_forward_reuse     = reverse_direction(Power_pump_backward);
            Power_pump_backward_reuse    = reverse_direction(Power_pump_forward);
            Power_ASE_forward_reuse      = reverse_direction(Power_ASE_backward);
            Power_ASE_backward_reuse     = reverse_direction(Power_ASE_forward);
            signal_fields_backward_reuse = reverse_direction(signal_fields);
            signal_fields_reuse          = signal_fields_backward_reuse; % dummy saved "signal_fields_reuse"
            [Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward,signal_fields,signal_fields_backward] = deal(Power_pump_forward_reuse,Power_pump_backward_reuse,Power_ASE_forward_reuse,Power_ASE_backward_reuse,signal_fields_reuse,signal_fields_backward_reuse);
        end
        if gain_rate_eqn.save_all_in_RAM
            saved_data.signal_fields          = signal_fields;
            saved_data.Power_pump_forward     = Power_pump_forward;
            saved_data.Power_pump_backward    = Power_pump_backward;
            saved_data.Power_ASE_forward      = Power_ASE_forward;
            saved_data.Power_ASE_backward     = Power_ASE_backward;
            saved_data.signal_fields_backward = signal_fields_backward;
        else
            save(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,1),'Power_pump_forward','Power_pump_backward','Power_ASE_forward','Power_ASE_backward','signal_fields','signal_fields_backward','-v7.3');
        end
    end
else
    % Initialization
    [signal_fields_out,signal_fields_backward_out,Power_pump_forward_out,Power_ASE_forward_out,Power_pump_backward_out,Power_ASE_backward_out] = initialization('both',sim,gain_rate_eqn,1,num_segments,N,num_modes,save_points,initial_condition);
    if sim.gpu_yes % "cell2mat" doesn't work for gpuArray
        [signal_fields_out,signal_fields_backward_out,Power_pump_forward_out,Power_pump_backward_out,Power_ASE_forward_out,Power_ASE_backward_out] = mygather(signal_fields_out,signal_fields_backward_out,Power_pump_forward_out,Power_pump_backward_out,Power_ASE_forward_out,Power_ASE_backward_out);
    end
    signal_fields_out          = cell2mat(signal_fields_out); % cell to array
    signal_fields_backward_out = cell2mat(signal_fields_backward_out);
    
    % Load the saved "mat" data and delete these temporary files.
    last_idx = 0;
    for segment_idx = 1:num_segments
        load(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,segment_idx),'Power_pump_forward','Power_pump_backward','Power_ASE_forward','Power_ASE_backward','signal_fields','signal_fields_backward');
        [signal_fields_segment,signal_fields_backward_segment,Power_pump_forward_segment,Power_pump_backward_segment,Power_ASE_forward_segment,Power_ASE_backward_segment] = deal(signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward);
        signal_fields_segment          = cell2mat(signal_fields_segment);
        signal_fields_backward_segment = cell2mat(signal_fields_backward_segment);
        
        % Change the size back to (N,num_modes).
        Power_ASE_forward_segment  = cellfun(@(P) permute(P,[5 3 1 2 4]), Power_ASE_forward_segment ,'UniformOutput',false);
        Power_ASE_backward_segment = cellfun(@(P) permute(P,[5 3 1 2 4]), Power_ASE_backward_segment,'UniformOutput',false);
        
        current_saved_z_points = saved_z_points((saved_z_points>(num_each_segment*(segment_idx-1))) & (saved_z_points<(num_each_segment*segment_idx+1)));
        if ~isempty(current_saved_z_points)
            current_saved_z_points_in_the_segment = rem(current_saved_z_points,num_each_segment);
            if current_saved_z_points_in_the_segment(end) == 0
                current_saved_z_points_in_the_segment(end) = num_each_segment;
            end
            current_save_idx = (1:length(current_saved_z_points)) + last_idx;

            Power_pump_forward_out(current_save_idx)  = Power_pump_forward_segment(current_saved_z_points_in_the_segment); % load Power
            Power_pump_backward_out(current_save_idx) = Power_pump_backward_segment(current_saved_z_points_in_the_segment);
            Power_ASE_forward_out(current_save_idx)   = Power_ASE_forward_segment(current_saved_z_points_in_the_segment);
            Power_ASE_backward_out(current_save_idx)  = Power_ASE_backward_segment(current_saved_z_points_in_the_segment);
            signal_fields_out(:,:,current_save_idx)   = fft(signal_fields_segment(:,:,current_saved_z_points_in_the_segment)); % load field
            signal_fields_backward_out(:,:,current_save_idx) = fft(signal_fields_backward_segment(:,:,current_saved_z_points_in_the_segment));
            
            if ~gain_rate_eqn.reuse_data
                delete(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,segment_idx));
            end
            last_idx = max(current_save_idx);
        end
    end
    signal_fields_out = struct('forward', signal_fields_out,...
                               'backward',signal_fields_backward_out);
    
    % Transform them into arrays
    Power_pump_forward_out  = cell2mat(Power_pump_forward_out);
    Power_pump_backward_out = cell2mat(Power_pump_backward_out);
    Power_ASE_forward_out   = fftshift(cell2mat(Power_ASE_forward_out ),1);
    Power_ASE_backward_out  = fftshift(cell2mat(Power_ASE_backward_out),1);
    Power_out = struct('pump_forward',Power_pump_forward_out,'pump_backward',Power_pump_backward_out,...
                       'ASE_forward', Power_ASE_forward_out, 'ASE_backward', Power_ASE_backward_out);
                   
    % Reverse the order and save the data for the linear oscillator scheme
    % for the next round
    if gain_rate_eqn.linear_oscillator
        if num_segments > 1
            tmp_i = 1;
            current_segment_idx = num_segments;
            saved_segment_idx = 1;
            reverse_direction = @(x) flip(x,3);
            
            for segment_idx = num_segments:-1:1
                tmp_saved_data(tmp_i) = load(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,segment_idx));
                if segment_idx == num_segments
                    save_idx = length(tmp_saved_data(tmp_i).Power_pump_forward);
                    if length(tmp_saved_data(tmp_i).Power_pump_forward) < num_each_segment
                        need_to_load_2nd_segment = true;
                        load_next_segment = false;
                    else
                        need_to_load_2nd_segment = false;
                        load_next_segment = false;
                    end
                end
                if need_to_load_2nd_segment && ~load_next_segment
                    tmp_i = 2;
                    load_next_segment = true;
                    continue;
                else
                    if need_to_load_2nd_segment
                        Power_pump_forward     = cell(1,1,num_each_segment);
                        Power_pump_backward    = cell(1,1,num_each_segment);
                        Power_ASE_forward      = cell(1,1,num_each_segment);
                        Power_ASE_backward     = cell(1,1,num_each_segment);
                        signal_fields_backward = cell(1,1,num_each_segment);
                    end
                    Power_pump_forward(1:save_idx)     = reverse_direction(tmp_saved_data(1).Power_pump_backward(1:save_idx));
                    Power_pump_backward(1:save_idx)    = reverse_direction(tmp_saved_data(1).Power_pump_forward(1:save_idx));
                    Power_ASE_forward(1:save_idx)      = reverse_direction(tmp_saved_data(1).Power_ASE_backward(1:save_idx));
                    Power_ASE_backward(1:save_idx)     = reverse_direction(tmp_saved_data(1).Power_ASE_forward(1:save_idx));
                    signal_fields_backward(1:save_idx) = reverse_direction(tmp_saved_data(1).signal_fields(1:save_idx));
                    if tmp_i == 2
                        Power_pump_forward(save_idx+1:num_each_segment)     = reverse_direction(tmp_saved_data(2).Power_pump_backward(save_idx+1:end));
                        Power_pump_backward(save_idx+1:num_each_segment)    = reverse_direction(tmp_saved_data(2).Power_pump_forward(save_idx+1:end));
                        Power_ASE_forward(save_idx+1:num_each_segment)      = reverse_direction(tmp_saved_data(2).Power_ASE_backward(save_idx+1:end));
                        Power_ASE_backward(save_idx+1:num_each_segment)     = reverse_direction(tmp_saved_data(2).Power_ASE_forward(save_idx+1:end));
                        signal_fields_backward(save_idx+1:num_each_segment) = reverse_direction(tmp_saved_data(2).signal_fields(save_idx+1:end));
                        
                        tmp_saved_data(1) = tmp_saved_data(2);
                    end
                    signal_fields = signal_fields_backward; % dummy saved "signal_fields"
                    save(sprintf('%s%u_.mat',gain_rate_eqn.saved_mat_filename,saved_segment_idx),'Power_pump_forward','Power_pump_backward','Power_ASE_forward','Power_ASE_backward','signal_fields','signal_fields_backward','-v7.3');
                    delete(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,current_segment_idx));
                    saved_segment_idx = saved_segment_idx + 1;
                    current_segment_idx = current_segment_idx - 1;
                    if segment_idx == 1 && need_to_load_2nd_segment
                        Power_pump_forward     = reverse_direction(tmp_saved_data(2).Power_pump_backward(1:save_idx));
                        Power_pump_backward    = reverse_direction(tmp_saved_data(2).Power_pump_forward(1:save_idx));
                        Power_ASE_forward      = reverse_direction(tmp_saved_data(2).Power_ASE_backward(1:save_idx));
                        Power_ASE_backward     = reverse_direction(tmp_saved_data(2).Power_ASE_forward(1:save_idx));
                        signal_fields_backward = reverse_direction(tmp_saved_data(2).signal_fields(1:save_idx));
                        signal_fields          = signal_fields_backward; % dummy saved "signal_fields"
                        save(sprintf('%s%u_.mat',gain_rate_eqn.saved_mat_filename,saved_segment_idx),'Power_pump_forward','Power_pump_backward','Power_ASE_forward','Power_ASE_backward','signal_fields','signal_fields_backward','-v7.3');
                        delete(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,current_segment_idx));
                    end
                end
            end
            % Rename the files
            for segment_idx = 1:num_segments
                movefile(sprintf('%s%u_.mat',gain_rate_eqn.saved_mat_filename,segment_idx),...
                         sprintf('%s%u.mat' ,gain_rate_eqn.saved_mat_filename,segment_idx));
            end
        end
    end
end

if ~gain_rate_eqn.linear_oscillator
    signal_fields_out = signal_fields_out.forward;
end

if gain_rate_eqn.reuse_data
    if gain_rate_eqn.save_all_in_RAM
        saved_data_info = saved_data;
    else
        saved_data_info = @()clean_saved_mat(gain_rate_eqn,num_segments);
    end
    if isequal(sim.step_method,'MPA') % MPA
        if gain_rate_eqn.export_N2
            varargout = {signal_fields_out,t_delay,Power_out,N2,full_iterations_hist,saved_data_info};
        else
            varargout = {signal_fields_out,t_delay,Power_out,   full_iterations_hist,saved_data_info};
        end
    else % split-stepp, RK4IP
        if gain_rate_eqn.export_N2
            varargout = {signal_fields_out,t_delay,Power_out,N2,saved_data_info};
        else
            varargout = {signal_fields_out,t_delay,Power_out,saved_data_info};
        end
    end
else
    if isequal(sim.step_method,'MPA') % MPA
        if gain_rate_eqn.export_N2
            varargout = {signal_fields_out,t_delay,Power_out,N2,full_iterations_hist};
        else
            varargout = {signal_fields_out,t_delay,Power_out,   full_iterations_hist};
        end
    else % split-stepp, RK4IP
        if gain_rate_eqn.export_N2
            varargout = {signal_fields_out,t_delay,Power_out,N2};
        else
            varargout = {signal_fields_out,t_delay,Power_out};
        end
    end
end

end

%%
function varargout = gain_propagate(direction,...
                                    sim,gain_rate_eqn,...
                                    z_points_all,segments,load_saved_files,save_points,num_large_steps_persave,...
                                    N,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,...
                                    cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN, ...
                                    initial_condition,...
                                    signal_fields,signal_fields_backward, ...
                                    Power_pump,Power_ASE)
%GAIN_PROPAGATE Runs the corresponding propagation method based on "direction".

dt = initial_condition.dt;

doesnt_need_iterations = isequal(gain_rate_eqn.pump_direction,'co') && gain_rate_eqn.ignore_ASE; % There's no need to compute backward propagation if copumping and ignoring ASE, so the first forward propagation solves everything.

num_each_segment = segments(1);
num_segments = length(segments);
num_modes = size(initial_condition.fields,2);

% The 1st backward-propagation for counter/bi-pumping which happens before iteration starts.
if isequal(direction,'backward') && ~load_saved_files
    first_backward_before_iterations = true;
else
    first_backward_before_iterations = false;
end

% Set up/initialize Power for the pump and ASE
if gain_rate_eqn.save_all_in_RAM && sim.gpu_yes
    z_points = z_points_all;
else
    switch direction
        case 'forward'
            z_points = segments(1);
        case 'backward'
            z_points = segments(end);
    end
end
switch direction
    case 'forward'
        Power_pump_backward = Power_pump;
        Power_ASE_backward  = Power_ASE;
        [signal_fields,Power_pump_forward,Power_ASE_forward] = initialization('forward',sim,gain_rate_eqn,1,num_segments,N,num_modes,z_points,initial_condition);
        if gain_rate_eqn.save_all_in_RAM && sim.gpu_yes
            [signal_fields,Power_pump_forward,Power_ASE_forward] = mygpuArray2(1,segments(1),signal_fields,Power_pump_forward,Power_ASE_forward);
        end
    case 'backward'
        % signal_fields is taken from the input argument
        Power_pump_forward = Power_pump;
        Power_ASE_forward  = Power_ASE;
        [Power_pump_backward,Power_ASE_backward] = initialization('backward',sim,gain_rate_eqn,num_segments,num_segments,N,num_modes,z_points,initial_condition);
        if gain_rate_eqn.save_all_in_RAM && sim.gpu_yes
            [Power_pump_backward,Power_ASE_backward] = mygpuArray2(z_points_all-segments(end)+1,z_points_all,Power_pump_backward,Power_ASE_backward);
        end
end

% Initialize N2 to be exported, the ion density of the upper state
if gain_rate_eqn.export_N2
    if sim.single_yes
        N2 = zeros([size(N_total) save_points],'single');
    else
        N2 = zeros([size(N_total) save_points]);
    end
end

% Setup a small matrix to track the number of iterations per step
if isequal(sim.step_method,'MPA') % MPA
    iterations_hist = zeros(sim.MPA.n_tot_max, 1);
    full_iterations_hist = zeros(sim.MPA.n_tot_max, save_points-1);
end

if sim.progress_bar
    if ~isfield(sim,'progress_bar_name')
        sim.progress_bar_name = '';
    elseif ~ischar(sim.progress_bar_name)
        error('GMMNLSE_propagate:ProgressBarNameError',...
            '"sim.progress_bar_name" should be a string.');
    end
    h_progress_bar = waitbar(0,sprintf('%s   0.0%%',sim.progress_bar_name),...
        'Name',sprintf('Running GMMNLSE (%s): %s...',direction,sim.progress_bar_name),...
        'CreateCancelBtn',...
        'setappdata(gcbf,''canceling'',1)');
    setappdata(h_progress_bar,'canceling',0);
    
    % Create the cleanup object
    cleanupObj = onCleanup(@()cleanMeUp(h_progress_bar));
    
    % Use this to control the number of updated time for the progress bar below 1000 times.
    count_progress_bar = 1;
    num_progress_updates = 1000;
end

t_delay = 0; % time delay

% Then start the propagation
for ii = 2:z_points_all
    % Check for Cancel button press
    if sim.progress_bar && getappdata(h_progress_bar,'canceling')
        if ~doesnt_need_iterations || gain_rate_eqn.reuse_data
            clean_saved_mat(gain_rate_eqn,num_segments);
        end
        
        error('GMMNLSE_propagate:ProgressBarBreak',...
        'The "cancel" button of the progress bar has been clicked.');
    end
    
    % =====================================================================
    % GMMNLSE: Run the correct step function depending on the options chosen.
    % =====================================================================
    switch direction
        case 'forward'
            Zi = ii;
            if gain_rate_eqn.save_all_in_RAM && sim.gpu_yes
                zi = Zi;
            else
                zi = rem(Zi,num_each_segment);
            end
            if zi == 0
                zi = num_each_segment;
            end
            
            if zi ~= 1
                last_signal_fields_backward = signal_fields_backward{zi-1};
                last_Power_pump_backward    = Power_pump_backward   {zi-1};
                last_Power_ASE_backward     = Power_ASE_backward    {zi-1};
                
                if Zi == 2 % the first/starting z_point
                    last_Power_pump_forward = Power_pump_forward{1};
                    last_Power_ASE_forward  = Power_ASE_forward {1};
                    last_signal_fields      = signal_fields     {1}; % = initial_condition.fields
                end
            end
            
            % For MPA, fundamental mode treats the gain as a dispersion term, 
            %    while multimode treats it as a nonlinear term due to the performance.
            % For split step, only fundamental-mode cases is available because it's hard to calculate the amplification for the gain-dispersion term if the field at a certain time/frequency is too small.
            % I don't implement the gain as a nonlinear term for the split-step algorithm because it just takes too much time to compute for the Runge-Kutta scheme.
            switch sim.step_method
                case 'split-step'
                    if isequal(gain_rate_eqn.midx,1) % fundamental mode
                        if gain_rate_eqn.export_N2
                            [       last_signal_fields,last_Power_pump_forward,last_Power_ASE_forward,N2_next] = GMMNLSE_ss_rategain(   last_signal_fields,last_signal_fields_backward,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,last_Power_ASE_backward,dt,sim,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,first_backward_before_iterations);
                        else
                            [       last_signal_fields,last_Power_pump_forward,last_Power_ASE_forward]         = GMMNLSE_ss_rategain(   last_signal_fields,last_signal_fields_backward,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,last_Power_ASE_backward,dt,sim,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,first_backward_before_iterations);
                        end
                    else % multimode or higher-order modes
                        error('GMMNLSE_gain_rate_eqn:AlgorithmError',...
                            'For multimode cases, there are only MPA and RK4IP available because there will be some problem with calculating the amplification for the dispersion term for the split-step algorithm.');
                    end
                case 'RK4IP'
                    if gain_rate_eqn.export_N2
                        [       last_signal_fields,last_Power_pump_forward,last_Power_ASE_forward,N2_next] = GMMNLSE_RK4IP_rategain(   last_signal_fields,last_signal_fields_backward,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,last_Power_ASE_backward,dt,sim,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,first_backward_before_iterations);
                    else
                        [       last_signal_fields,last_Power_pump_forward,last_Power_ASE_forward]         = GMMNLSE_RK4IP_rategain(   last_signal_fields,last_signal_fields_backward,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,last_Power_ASE_backward,dt,sim,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,first_backward_before_iterations);
                    end
                case 'MPA'
                    if isequal(gain_rate_eqn.midx,1) % fundamental mode
                        if gain_rate_eqn.export_N2
                            [num_it,last_signal_fields,last_Power_pump_forward,last_Power_ASE_forward,N2_next] = GMMNLSE_MPA_SMrategain(last_signal_fields,last_signal_fields_backward,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,last_Power_ASE_backward,dt,sim,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,first_backward_before_iterations);
                        else
                            [num_it,last_signal_fields,last_Power_pump_forward,last_Power_ASE_forward]         = GMMNLSE_MPA_SMrategain(last_signal_fields,last_signal_fields_backward,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,last_Power_ASE_backward,dt,sim,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,first_backward_before_iterations);
                        end
                    else % multimode or higher-order modes
                        if gain_rate_eqn.export_N2
                            [num_it,last_signal_fields,last_Power_pump_forward,last_Power_ASE_forward,N2_next] = GMMNLSE_MPA_MMrategain(last_signal_fields,last_signal_fields_backward,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,last_Power_ASE_backward,dt,sim,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,first_backward_before_iterations);
                        else
                            [num_it,last_signal_fields,last_Power_pump_forward,last_Power_ASE_forward]         = GMMNLSE_MPA_MMrategain(last_signal_fields,last_signal_fields_backward,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,last_Power_ASE_backward,dt,sim,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,first_backward_before_iterations);
                        end
                    end
                    iterations_hist(num_it) = iterations_hist(num_it)+1;
            end
            
            % Update "forward" only or
            % Save them if it doesn't need iterations.
            if doesnt_need_iterations && ~gain_rate_eqn.reuse_data
                if rem(Zi-1, num_large_steps_persave) == 0
                    Power_pump_forward{int64((Zi-1)/num_large_steps_persave+1)} = last_Power_pump_forward;
                    Power_ASE_forward {int64((Zi-1)/num_large_steps_persave+1)} = last_Power_ASE_forward;
                    signal_fields     {int64((Zi-1)/num_large_steps_persave+1)} = last_signal_fields;
                end
            else
                Power_pump_forward{zi} = last_Power_pump_forward;
                Power_ASE_forward {zi} = last_Power_ASE_forward;
                signal_fields     {zi} = last_signal_fields;
            end
            
            % Save N2
            if gain_rate_eqn.export_N2
                if Zi == 2 % Put the first N2
                    if sim.gpu_yes
                        N2(:,:,1) = gather(N2_next(:,:,1,1,1,end));
                    else
                        N2(:,:,1) = N2_next(:,:,1,1,1,end);
                    end
                end
                if rem(Zi-1, num_large_steps_persave) == 0
                    if sim.gpu_yes % if using MPA, save only the last one
                        N2(:,:,int64((Zi-1)/num_large_steps_persave+1)) = gather(N2_next(:,:,1,1,1,end));
                    else
                        N2(:,:,int64((Zi-1)/num_large_steps_persave+1)) = N2_next(:,:,1,1,1,end);
                    end
                end
            end
            
            % Save Power and signal_fields to a file
            if rem(zi,num_each_segment) == 0 && num_segments ~= 1
                save_files = true;
            else
                save_files = false;
            end
            
        case 'backward'
            Zi = z_points_all+1 - ii;
            if gain_rate_eqn.save_all_in_RAM && sim.gpu_yes
                zi = Zi;
            else
                zi = rem(Zi,num_each_segment);
            end
            if zi == 0
                zi = num_each_segment;
                
                % If unfortunately ii=2 corresponds to the last z_point in 
                % a segment, the last power and signal fields are taken
                % directly from Power and signal_fields because they aren't
                % loaded in advance below.
                if Zi == z_points_all-1 % the last/starting z_point
                    last_Power_pump_forward     = Power_pump_forward {:};
                    last_Power_pump_backward    = Power_pump_backward{:};
                    last_Power_ASE_forward      = Power_ASE_forward  {:};
                    last_Power_ASE_backward     = Power_ASE_backward {:};
                    
                    last_signal_fields          = signal_fields         {:};
                    last_signal_fields_backward = signal_fields_backward{:};
                end
            elseif zi ~= num_each_segment
                last_Power_pump_forward     = Power_pump_forward {zi+1};
                last_Power_pump_backward    = Power_pump_backward{zi+1};
                last_Power_ASE_forward      = Power_ASE_forward  {zi+1};
                last_Power_ASE_backward     = Power_ASE_backward {zi+1};
                
                last_signal_fields          = signal_fields         {zi+1};
                last_signal_fields_backward = signal_fields_backward{zi+1};
            end
            if isequal(sim.step_method,'MPA') % MPA
                deltaZ = sim.MPA.M*sim.deltaZ;
            else % split-step, RK4IP
                deltaZ = sim.deltaZ;
            end
            
            [last_Power_pump_backward,last_Power_ASE_backward,N2_next] = solve_gain_rate_eqn('backward',sim,gain_rate_eqn,last_signal_fields,last_signal_fields_backward,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,last_Power_ASE_backward,omegas,dt,deltaZ,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,first_backward_before_iterations);
            
            % Update "backward" only
            Power_pump_backward{zi} = last_Power_pump_backward;
            Power_ASE_backward {zi} = last_Power_ASE_backward;
            
            % Save Power and signal_fields to a file
            if rem(zi,num_each_segment) == 1 && Zi ~= 1
                save_files = true;
            else
                save_files = false;
            end
    end
    
    % =====================================================================
    % Others
    % =====================================================================
    if isequal(direction,'forward')
        % Check for any NaN elements, if desired
        if sim.check_nan
            if any(any(isnan(last_signal_fields))) %any(isnan(last_signal_fields),'all')
                if ~doesnt_need_iterations || gain_rate_eqn.reuse_data || ~gain_rate_eqn.save_all_in_RAM
                    clean_saved_mat(gain_rate_eqn,num_segments);
                end
                
                error('GMMNLSE_propagate:NaNError',...
                    'NaN field encountered, aborting.\nPossible reason is that the nonlinear length is too close to the large step size, deltaZ*M for MPA or deltaZ for split-step.');
            end
        end
        
        % Center the pulse Pavel commet
%         last_signal_fields_in_time = fft(last_signal_fields);
%         tCenter = floor(sum(sum((-floor(N/2):floor((N-1)/2))'.*abs(last_signal_fields_in_time).^2),2)/sum(sum(abs(last_signal_fields_in_time).^2),2));
%         if ~isnan(tCenter) && tCenter ~= 0 % all-zero fields; for calculating ASE power only
%             % Because circshift is slow on GPU, I discard it.
%             %last_signal_fields = ifft(circshift(last_signal_fields_in_time,-tCenter));
%             if tCenter > 0
%                 last_signal_fields = ifft([last_signal_fields_in_time(1+tCenter:end,:);last_signal_fields_in_time(1:tCenter,:)]);
%             elseif tCenter < 0
%                 last_signal_fields = ifft([last_signal_fields_in_time(end+1+tCenter:end,:);last_signal_fields_in_time(1:end+tCenter,:)]);
%             end
%             if sim.gpu_yes
%                 t_delay = t_delay + gather(tCenter*dt);
%             else
%                 t_delay = t_delay + tCenter*dt;
%             end
%         end
    end
    
    % Save and load files
    if save_files
        current_segment_idx = ceil(Zi/num_each_segment);
        switch direction
            case 'forward'
                next_segment_idx = current_segment_idx + 1;
            case 'backward'
                next_segment_idx = current_segment_idx - 1;
        end
        if ~doesnt_need_iterations || gain_rate_eqn.reuse_data
            last_Power_pump_forward  = Power_pump_forward {zi};
            last_Power_pump_backward = Power_pump_backward{zi};
            last_Power_ASE_forward   = Power_ASE_forward  {zi};
            last_Power_ASE_backward  = Power_ASE_backward {zi};
            last_signal_fields       = signal_fields      {zi};
            
            if gain_rate_eqn.save_all_in_RAM && sim.gpu_yes
                cumsum_segments = [0 cumsum(segments)];
                starti = cumsum_segments(current_segment_idx)+1;
                endi   = cumsum_segments(current_segment_idx+1);
                [signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward] = mygather2(starti,endi,signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward);
            else
                if sim.gpu_yes
                    [signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward] = mygather(signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward);
                end
                save(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,current_segment_idx),'Power_pump_forward','Power_pump_backward','Power_ASE_forward','Power_ASE_backward','signal_fields','signal_fields_backward','-v7.3');
            end
            
            if next_segment_idx <= length(segments) % if z_points_all=num_each_segments*length(segments) exactly, next_segment_idx can be larger than length(segments). A check is provided here.
                if load_saved_files
                    if gain_rate_eqn.save_all_in_RAM && sim.gpu_yes
                        starti = cumsum_segments(next_segment_idx)+1;
                        endi   = cumsum_segments(next_segment_idx+1);
                        [signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward] = mygpuArray2(starti,endi,signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward);
                    else
                        load(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,next_segment_idx),'Power_pump_forward','Power_pump_backward','Power_ASE_forward','Power_ASE_backward','signal_fields','signal_fields_backward');
                        if sim.gpu_yes
                            [signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward] = mygpuArray(signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward);
                        end
                    end
                else % the 1st run of forward or backward propagation (hense no saved data yet)
                    if gain_rate_eqn.save_all_in_RAM && sim.gpu_yes
                        starti = cumsum_segments(next_segment_idx)+1;
                        endi   = cumsum_segments(next_segment_idx+1);
                        [signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward] = mygpuArray2(starti,endi,signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward);
                    else
                        [signal_fields,signal_fields_backward,Power_pump_forward,Power_ASE_forward,Power_pump_backward,Power_ASE_backward] = initialization('both',sim,gain_rate_eqn,next_segment_idx,length(segments),N,num_modes,segments(next_segment_idx),initial_condition);
                    end
                end
            end
        end
    end
    
    % Report current status in the progress bar's message field
    if sim.progress_bar
        if z_points_all < num_progress_updates || floor((ii-1)/((z_points_all-1)/num_progress_updates)) == count_progress_bar
            waitbar((ii-1)/(z_points_all-1),h_progress_bar,sprintf('%s%6.1f%%',sim.progress_bar_name,(ii-1)/(z_points_all-1)*100));
            count_progress_bar = count_progress_bar+1;
        end
    end
end
% When the data needs to be used later,
% if num_segments=1, it'll be saved in GMMNLSE_gain_rate_eqn, the caller function of this sub-function above;
% if num_segments>1, the latest one will be saved below.
if isequal(direction,'forward')
    if ~gain_rate_eqn.save_all_in_RAM || ~sim.gpu_yes
        if num_segments ~= 1 && (~doesnt_need_iterations || gain_rate_eqn.reuse_data)
            if sim.gpu_yes
                [signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward] = mygather(signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward);
            end
            save(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,num_segments),'Power_pump_forward','Power_pump_backward','Power_ASE_forward','Power_ASE_backward','signal_fields','signal_fields_backward','-v7.3');
            if sim.gpu_yes
                [signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward] = mygpuArray(signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward);
            end
        end
    end
end

% Output
switch direction
    case 'forward'
        if isequal(sim.step_method,'MPA') % MPA
            if gain_rate_eqn.export_N2
                N2 = prep_for_output_N2(N2,N_total,num_large_steps_persave);
                varargout = {signal_fields_backward,t_delay,Power_pump_forward,Power_ASE_forward,signal_fields,N2,full_iterations_hist};
            else
                varargout = {signal_fields_backward,t_delay,Power_pump_forward,Power_ASE_forward,signal_fields,   full_iterations_hist};
            end
        else % split-stepp, RK4IP
            if gain_rate_eqn.export_N2
                N2 = prep_for_output_N2(N2,N_total,num_large_steps_persave);
                varargout = {signal_fields_backward,t_delay,Power_pump_forward,Power_ASE_forward,signal_fields,N2};
            else
                varargout = {signal_fields_backward,t_delay,Power_pump_forward,Power_ASE_forward,signal_fields};
            end
        end
    case 'backward'
        varargout = {signal_fields_backward,Power_pump_backward,Power_ASE_backward};
end

end

%% initialization
function varargout = initialization(direction,sim,gain_rate_eqn,segment_idx,num_segments,N,num_modes,z_points,initial_condition)
%INITIALIZATION initializes "signal_fields" and "Power" based on
%"segment_idx/num_segment".
%
%   If segment_idx = 1, they need to include copump_power, initial_fields, and initial forward ASE.
%   If segment_idx = num_segment(the last segment), "Power" needs to include counterpump_power if it's nonzero.
%                                                   They also need to include initial backward ASE.
%   Otherwise, they're just a zero matrix and a structure with zero matrices.
%
%   The reason I use cell arrays instead of a matrix (N,num_modes,z_points) for signal_fields and Power:
%       It's faster!
%       e.g. "signal_fields(:,:,zi) = signal_fields_next" is very slow.

    function output = initialize_zeros(mat_size,single_yes)
        output = cell(1,1,z_points);
        if single_yes
            output(:) = {zeros(mat_size,'single')};
        else
            output(:) = {zeros(mat_size)};
        end
    end

% Power
% "cell2mat" doesn't support "gpuArray" in a cell array, which affects the process when getting the output matrix. 
% Pump
if any(strcmp(direction,{'forward','both'}))
    Power_pump_forward = initialize_zeros(1,sim.single_yes);
end
if any(strcmp(direction,{'backward','both'}))
    Power_pump_backward = initialize_zeros(1,sim.single_yes);
end
% ASE
if any(strcmp(direction,{'forward','both'}))
    if gain_rate_eqn.ignore_ASE % Because ASE is ignored, set it a scalar zero is enough.
        Power_ASE_forward = initialize_zeros(1,sim.single_yes);
    else
        % Make it the size to (1,1,num_modes,1,N) for "solve_gain_rate_eqn.m"
        Power_ASE_forward = initialize_zeros([1,1,num_modes,1,N],sim.single_yes);
    end
end
if any(strcmp(direction,{'backward','both'}))
    if gain_rate_eqn.ignore_ASE % Because ASE is ignored, set it a scalar zero is enough.
        Power_ASE_backward = initialize_zeros(1,sim.single_yes);
    else
        % Make it the size to (1,1,num_modes,1,N) for "solve_gain_rate_eqn.m"
        Power_ASE_backward = initialize_zeros([1,1,num_modes,1,N],sim.single_yes);
    end
end

% Put in the necessary information.
% Pump power
if any(strcmp(direction,{'forward','both'}))
    if segment_idx == 1 && any(strcmp(gain_rate_eqn.pump_direction,{'co','bi'}))
        if sim.single_yes
            Power_pump_forward{1} = single(gain_rate_eqn.copump_power);
        else
            Power_pump_forward{1} = gain_rate_eqn.copump_power;
        end
    end
end
if any(strcmp(direction,{'backward','both'}))
    if segment_idx == num_segments && any(strcmp(gain_rate_eqn.pump_direction,{'counter','bi'}))
        if sim.single_yes
            Power_pump_backward{end} = single(gain_rate_eqn.counterpump_power);
        else
            Power_pump_backward{end} = gain_rate_eqn.counterpump_power;
        end
    end
end
% ASE
if any(strcmp(direction,{'forward','both'}))
    if ~gain_rate_eqn.ignore_ASE
        % Make it the size to (1,1,num_modes,1,N) for "solve_gain_rate_eqn.m"
        if sim.single_yes
            Power_ASE_forward{1} = single(permute(initial_condition.ASE_forward,[3 4 2 5 1]));
        else
            Power_ASE_forward{1} = permute(initial_condition.ASE_forward,[3 4 2 5 1]);
        end
    end
end
if any(strcmp(direction,{'backward','both'}))
    if ~gain_rate_eqn.ignore_ASE
        % Make it the size to (1,1,num_modes,1,N) for "solve_gain_rate_eqn.m"
        if sim.single_yes
            Power_ASE_backward{end} = single(permute(initial_condition.ASE_backward,[3 4 2 5 1]));
        else
            Power_ASE_backward{end} = permute(initial_condition.ASE_backward,[3 4 2 5 1]);
        end
    end
end

% -------------------------------------------------------------------------
% Signal field
% "cell2mat" doesn't support "gpuArray" in a cell array, which affects the process when getting the output matrix. 
if any(strcmp(direction,{'forward','both'}))
    signal_fields = initialize_zeros([N,num_modes],sim.single_yes);
    if segment_idx == 1
        if sim.single_yes
            signal_fields{1} = single(initial_condition.fields);
        else
            signal_fields{1} = initial_condition.fields;
        end
    end
end
% This signal_fields_backward is used only for linear oscillator scheme.
if isequal(direction,'both')
    signal_fields_backward = initialize_zeros(1,sim.single_yes);
end

% -------------------------------------------------------------------------
% GPU
if sim.gpu_yes && ~gain_rate_eqn.save_all_in_RAM
    if any(strcmp(direction,{'forward','both'}))
        [signal_fields,Power_pump_forward,Power_ASE_forward] = mygpuArray(signal_fields,Power_pump_forward,Power_ASE_forward);
    end
    if any(strcmp(direction,{'backward','both'}))
        [Power_pump_backward,Power_ASE_backward] = mygpuArray(Power_pump_backward,Power_ASE_backward);
    end
    if strcmp(direction,'both')
        signal_fields_backward = cellfun(@gpuArray,signal_fields_backward,'UniformOutput',false);
    end
end

switch direction
    case 'forward'
        varargout = {signal_fields,Power_pump_forward,Power_ASE_forward};
    case 'backward'
        varargout = {Power_pump_backward,Power_ASE_backward};
    otherwise % 'both'
        varargout = {signal_fields,signal_fields_backward,Power_pump_forward,Power_ASE_forward,Power_pump_backward,Power_ASE_backward};
end

end

%% CALC_TOTAL_ENERGY
function total_energy = calc_total_energy(A,N,dt)
%total_energy = sum(trapz(abs(fftshift(A,1)).^2))*N*dt/1e3; % in nJ

center_idx = ceil(size(A,1)/2);
A2 = abs(A).^2;

total_energy = (sum(sum(A2))-sum(sum(A2([center_idx,center_idx+1],:))))*N*dt/1e3; % in nJ;

end

%% PREP_FOR_OUTPUT_N2
function N2 = prep_for_output_N2( N2, N_total, num_large_steps_persave )
%PREP_FOR_OUTPUT_N2 It interpolates the first z-plane N2 from the other N2
%data and transform N2 into the ratio, N2/N_total.

if isequal(class(N_total),'gpuArray')
    N_total = gather(N_total);
end

N2 = N2/max(N_total(:));
sN2 = size(N2,3)-1;

if sN2 > 1
    N2(:,:,1) = permute(interp1([1,(1:sN2)*num_large_steps_persave],permute(N2,[3 1 2]),0,'spline'),[2 3 1]);
end

end

%% CLEAN_SAVED_MAT
function clean_saved_mat(gain_rate_eqn,num_segments)
%CLEAN_SAVED_MAT It deletes the saved mat files for iterations.

for segment_idx = 1:num_segments
    current_caller_path = cd;
    if (exist(fullfile(current_caller_path, sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,segment_idx)), 'file') == 2)
        delete( fullfile(current_caller_path, sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,segment_idx)) );
    end
end

end

%% MYGPUARRAY
function varargout = mygpuArray(varargin)
%MYGPUARRAY

varargout = cell(1,nargin);
for i = 1:nargin
    varargout{i} = cellfun(@gpuArray,varargin{i},'UniformOutput',false);
end

end

%% MYGATHER
function varargout = mygather(varargin)
%MYGATHER

varargout = cell(1,nargin);
for i = 1:nargin
    varargout{i} = cellfun(@gather,varargin{i},'UniformOutput',false);
end

end

%% MYGPUARRAY2
function varargout = mygpuArray2(starti,endi,varargin)
%MYGPUARRAY2

varargout = varargin;
for i = 1:nargin-2
    x = cellfun(@gpuArray,varargin{i}(starti:endi),'UniformOutput',false);
    varargout{i}(starti:endi) = x;
end

end

%% MYGATHER2
function varargout = mygather2(starti,endi,varargin)
%MYGATHER2

varargout = varargin;
for i = 1:nargin-2
    x = cellfun(@gather,varargin{i}(starti:endi),'UniformOutput',false);
    varargout{i}(starti:endi) = x;
end

end

%% CLEANMEUP
% -------------------------------------------------------------------------
function cleanMeUp(h_progress_bar)
%CLEANMEUP It deletes the progress bar.

% DELETE the progress bar; don't try to CLOSE it.
delete(h_progress_bar);
    
end
% -------------------------------------------------------------------------