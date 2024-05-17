function varargout = GMMNLSE_gain_rate_eqn_linear_oscillator( sim,gain_rate_eqn,...
                                                              cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,...
                                                              z_points_all,save_points,num_large_steps_persave,...
                                                              initial_condition,...
                                                              prefactor, SRa_info, SRb_info, SK_info, omegas, D_op, haw, hbw,...
                                                              saved_data)
%GMMNLSE_GAIN_RATE_EQN_LINEAR_OSCILLATOR It attains the field after 
%propagation inside the gain medium solved by the rate equations.
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
%
% Output:
%   signal_fields_out - the signal field; (N,num_modes,save_points)
%   Power_out - a structure with
%               pump_forward - (1,1,save_points)
%               pump_backward - (1,1,save_points)
%               ASE_forward - (N,num_modes,save_points)
%               ASE_backward - (N,num_modes,save_points)
%   N2 - the ion density in the upper state; (Nx,Nx,save_points)
%        the output of N2 is transformed into N2/N_total, the ratio of
%        population inversion
%   full_iterations_hist - histogram of the number of iterations, accumulated and saved between each save point
%   clean_rate_gain_mat - the function handle to delete all the saved mat files

N = size(initial_condition.fields,1);
num_modes = size(initial_condition.fields,2);
if sim.single_yes
    save_points = single(save_points);
else
    save_points = double(save_points);
end

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

%% To consider the iterations between forward and backward propagations, "all the information" about signal_fields and Power are saved to "mat" files to save the memory.
% These files will be consistently updated and loaded throughout iterations.
% For more details about the iteration process, please refer to 
% "Lindberg et al., "Accurate modeling of high-repetition rate ultrashort pulse amplification in optical fibers", Scientific Reports (2016)"

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
if gain_rate_eqn.linear_oscillator
    variable_size.signal_fields = 2*N*num_modes*z_points_all; % forward and backward
else
    variable_size.signal_fields = N*num_modes*z_points_all;
end
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

%% Propagations
if gain_rate_eqn.save_all_in_RAM
    Power_pump_backward    = saved_data.Power_pump_backward;
    Power_ASE_backward     = saved_data.Power_ASE_backward;
    signal_fields_backward = saved_data.signal_fields_backward;
else
    load(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,1),'Power_pump_backward','Power_ASE_backward','signal_fields_backward');
end
Power_pump_backward = extract_cell_content(Power_pump_backward); % It's sent into GPU right after or during initialization process
if sim.gpu_yes
    Power_ASE_backward     = mygpuArray2(1,segments(1),Power_ASE_backward);
    signal_fields_backward = mygpuArray2(1,segments(1),signal_fields_backward);
end

% Start the pulse propagation.
if isequal(sim.step_method,'MPA')
    if gain_rate_eqn.export_N2
        [signal_fields,signal_fields_backward,t_delay,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward,N2,full_iterations_hist] = gain_propagate(sim,gain_rate_eqn,z_points_all,segments,save_points,num_large_steps_persave,N,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,initial_condition,Power_pump_backward,signal_fields_backward,Power_ASE_backward);
    else
        [signal_fields,signal_fields_backward,t_delay,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward,   full_iterations_hist] = gain_propagate(sim,gain_rate_eqn,z_points_all,segments,save_points,num_large_steps_persave,N,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,initial_condition,Power_pump_backward,signal_fields_backward,Power_ASE_backward);
    end
else % split-step, RK4IP
    if gain_rate_eqn.export_N2
        [signal_fields,signal_fields_backward,t_delay,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward,N2]                      = gain_propagate(sim,gain_rate_eqn,z_points_all,segments,save_points,num_large_steps_persave,N,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,initial_condition,Power_pump_backward,signal_fields_backward,Power_ASE_backward);
    else
        [signal_fields,signal_fields_backward,t_delay,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward]                         = gain_propagate(sim,gain_rate_eqn,z_points_all,segments,save_points,num_large_steps_persave,N,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,initial_condition,Power_pump_backward,signal_fields_backward,Power_ASE_backward);
    end
end

% Output:
% Power_out is in frequency domain while field_out is transformed into time domain.
saved_z_points = 1:num_large_steps_persave:z_points_all;
if num_segments == 1 || (gain_rate_eqn.save_all_in_RAM && sim.gpu_yes)
    % Transform the current "signal_fields" and "Power..." into arrays.
    if sim.gpu_yes
        [signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward] = mygather(signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward);
    end
    signal_fields_backward_out = cell2mat(signal_fields_backward(1,1,saved_z_points));
    signal_fields_out          = cell2mat(signal_fields         (:,:,saved_z_points));
    % Change the size back to (N,num_modes)
    Power_ASE_forward_out   = cellfun(@(P) permute(P,[5 3 1 2 4]), Power_ASE_forward (:,:,saved_z_points),'UniformOutput',false);
    Power_ASE_backward_out  = cellfun(@(P) permute(P,[5 3 1 2 4]), Power_ASE_backward(:,:,saved_z_points),'UniformOutput',false);

    % Transform them into arrays
    Power_pump_forward_out  = cell2mat(Power_pump_forward);
    Power_pump_backward_out = cell2mat(Power_pump_backward);
    Power_ASE_forward_out   = fftshift(cell2mat(Power_ASE_forward_out ),1);
    Power_ASE_backward_out  = fftshift(cell2mat(Power_ASE_backward_out),1);   
    
    Power_out = struct('pump_forward',Power_pump_forward_out,'pump_backward',Power_pump_backward_out,...
                       'ASE_forward', Power_ASE_forward_out, 'ASE_backward', Power_ASE_backward_out);
    signal_fields_out = struct('forward', fft(signal_fields_out),...
                               'backwrad',fft(signal_fields_backward_out));
    
    % Reverse the order and save the data for the linear oscillator scheme
    % for the next round
    Power_pump_backward = Power_pump_forward{end};
    Power_ASE_backward  = flip(Power_ASE_forward,3);
    signal_fields_backward = flip(signal_fields,3);
    if gain_rate_eqn.save_all_in_RAM
        saved_data.Power_pump_backward    = Power_pump_backward;
        saved_data.Power_ASE_backward     = Power_ASE_backward;
        saved_data.signal_fields_backward = signal_fields_backward;
    else
        save(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,1),'Power_pump_backward','Power_ASE_backward','signal_fields_backward','-v7.3');
    end
else
    % Initialization
    if sim.gpu_yes
        initial_condition = structfun(@gather,initial_condition);
    end
    if sim.single_yes
        Power_ASE_forward_out  = zeros(N,num_modes,length(saved_z_points),'single');
        Power_ASE_backward_out = zeros(N,num_modes,length(saved_z_points),'single');
        signal_fields_out = zeros(N,num_modes,length(saved_z_points),'single');
        signal_fields_backward_out = zeros(N,num_modes,length(saved_z_points),'single');
        signal_fields_out(:,:,1) = single(initial_condition.fields);
        if ~gain_rate_eqn.ignore_ASE
            Power_ASE_forward_out(:,:,1) = single(initial_condition.ASE_forward);
        end
    else
        Power_ASE_forward_out  = zeros(N,num_modes,length(saved_z_points));
        Power_ASE_backward_out = zeros(N,num_modes,length(saved_z_points));
        signal_fields_out = zeros(N,num_modes,length(saved_z_points));
        signal_fields_backward_out = zeros(N,num_modes,length(saved_z_points));
        signal_fields_out(:,:,1) = initial_condition.fields;
        if ~gain_rate_eqn.ignore_ASE
            Power_ASE_forward_out(:,:,1) = initial_condition.ASE_forward;
        end
    end
    
    % Load the saved "mat" data and delete these temporary files.
    last_idx = 0;
    for segment_idx = 1:num_segments
        load(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,segment_idx),'signal_fields','signal_fields_backward','Power_ASE_forward','Power_ASE_backward');
        signal_fields_segment          = cell2mat(signal_fields);
        signal_fields_backward_segment = cell2mat(signal_fields_backward);
        
        Power_ASE_forward_segment  = Power_ASE_forward;
        Power_ASE_backward_segment = Power_ASE_backward;
        % Change the size back to (N,num_modes).
        Power_ASE_forward_segment = cellfun(@(P) permute(P,[5 3 1 2 4]), Power_ASE_forward_segment,'UniformOutput',false);
        Power_ASE_forward_segment = cell2mat(Power_ASE_forward_segment);
        Power_ASE_backward_segment = cellfun(@(P) permute(P,[5 3 1 2 4]), Power_ASE_backward_segment,'UniformOutput',false);
        Power_ASE_backward_segment = cell2mat(Power_ASE_backward_segment);
        
        current_saved_z_points = saved_z_points((saved_z_points>(num_each_segment*(segment_idx-1))) & (saved_z_points<(num_each_segment*segment_idx+1)));
        if ~isempty(current_saved_z_points)
            current_saved_z_points_in_the_segment = rem(current_saved_z_points,num_each_segment);
            if current_saved_z_points_in_the_segment(end) == 0
                current_saved_z_points_in_the_segment(end) = num_each_segment;
            end
            current_save_idx = (1:length(current_saved_z_points)) + last_idx;
            
            Power_ASE_forward_out(:,:,current_save_idx) = Power_ASE_forward_segment(:,:,current_saved_z_points_in_the_segment);
            Power_ASE_backward_out(:,:,current_save_idx) = Power_ASE_backward_segment(:,:,current_saved_z_points_in_the_segment);
            signal_fields_out(:,:,current_save_idx) = fft(signal_fields_segment(:,:,current_saved_z_points_in_the_segment)); % load field
            signal_fields_backward_out(:,:,current_save_idx) = fft(signal_fields_backward_segment(:,:,current_saved_z_points_in_the_segment));

            last_idx = max(current_save_idx);
        end
    end
    signal_fields_out = struct('forward', signal_fields_out,...
                               'backward',signal_fields_backward_out);
    
    % Transform them into arrays
    Power_pump_forward_out  = cell2mat(Power_pump_forward);
    Power_pump_backward_out = cell2mat(Power_pump_backward);
    Power_out = struct('pump_forward',Power_pump_forward_out,            'pump_backward',Power_pump_backward_out,...
                       'ASE_forward', fftshift(Power_ASE_forward_out,1), 'ASE_backward', fftshift(Power_ASE_backward_out,1));
                   
    % Reverse the order and save the data for the linear oscillator scheme
    % for the next round
    tmp_i = 1;
    current_segment_idx = num_segments;
    saved_segment_idx = 1;
    reverse_direction = @(x) flip(x,3);

    for segment_idx = num_segments:-1:1
        tmp_saved_data(tmp_i) = load(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,segment_idx),'signal_fields','Power_ASE_forward');
        if segment_idx == num_segments
            save_idx = size(tmp_saved_data(tmp_i).signal_fields,3);
            if size(tmp_saved_data(tmp_i).signal_fields,3) < num_each_segment
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
                Power_ASE_backward     = cell(1,1,num_each_segment);
                signal_fields_backward = cell(1,1,num_each_segment);
            end
            Power_ASE_backward    (1:save_idx) = reverse_direction(tmp_saved_data(1).Power_ASE_forward(1:save_idx));
            signal_fields_backward(1:save_idx) = reverse_direction(tmp_saved_data(1).signal_fields    (1:save_idx));
            if tmp_i == 2
                Power_ASE_backward    (save_idx+1:num_each_segment) = reverse_direction(tmp_saved_data(2).Power_ASE_forward(save_idx+1:end));
                signal_fields_backward(save_idx+1:num_each_segment) = reverse_direction(tmp_saved_data(2).signal_fields    (save_idx+1:end));

                tmp_saved_data(1) = tmp_saved_data(2);
            end
            save(sprintf('%s%u_.mat',gain_rate_eqn.saved_mat_filename,saved_segment_idx),'signal_fields_backward','Power_ASE_backward','-v7.3');
            delete(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,current_segment_idx));
            saved_segment_idx = saved_segment_idx + 1;
            current_segment_idx = current_segment_idx - 1;
            if segment_idx == 1 && need_to_load_2nd_segment
                Power_ASE_backward = reverse_direction(tmp_saved_data(2).Power_ASE_forward(1:save_idx));
                signal_fields_backward = reverse_direction(tmp_saved_data(2).signal_fields(1:save_idx));
                save(sprintf('%s%u_.mat',gain_rate_eqn.saved_mat_filename,saved_segment_idx),'signal_fields_backward','Power_ASE_backward','-v7.3');
                delete(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,current_segment_idx));
            end
        end
    end
    % Rename the files
    for segment_idx = 1:num_segments
        movefile(sprintf('%s%u_.mat',gain_rate_eqn.saved_mat_filename,segment_idx),...
                 sprintf('%s%u.mat' ,gain_rate_eqn.saved_mat_filename,segment_idx));
    end
    Power_pump_backward = Power_pump_forward{end};
    save(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,1),'Power_pump_backward','-append');
end

if gain_rate_eqn.save_all_in_RAM
    saved_data_info = saved_data;
else
    saved_data_info = @()clean_saved_mat(gain_rate_eqn,num_segments);
end
if isequal(sim.step_method,'MPA')
    if gain_rate_eqn.export_N2
        varargout = {signal_fields_out,t_delay,Power_out,N2,full_iterations_hist,saved_data_info};
    else
        varargout = {signal_fields_out,t_delay,Power_out,   full_iterations_hist,saved_data_info};
    end
else % split-step, RK4IP
    if gain_rate_eqn.export_N2
        varargout = {signal_fields_out,t_delay,Power_out,N2,saved_data_info};
    else
        varargout = {signal_fields_out,t_delay,Power_out,saved_data_info};
    end
end

end

%%
function varargout = gain_propagate(sim,gain_rate_eqn,...
                                    z_points_all,segments,save_points,num_large_steps_persave,...
                                    N,prefactor, SRa_info, SRb_info, SK_info, omegas, D_op, haw, hbw,...
                                    cross_sections_pump, cross_sections, overlap_factor, N_total, FmFnN, GammaN, ...
                                    initial_condition, input_Power_pump_backward, ...
                                    signal_fields_backward, Power_ASE_backward)
%GAIN_PROPAGATE Runs the corresponding propagation method.

dt = initial_condition.dt;

num_each_segment = segments(1);
num_segments = length(segments);
num_modes = size(initial_condition.fields,2);

% Set up/initialize Power for the pump and ASE
if gain_rate_eqn.save_all_in_RAM && sim.gpu_yes
    z_points = z_points_all;
else
    z_points = segments(1);
end
[signal_fields,Power_pump_forward,Power_pump_backward,Power_ASE_forward] = initialization(sim,gain_rate_eqn,z_points,N,num_modes,save_points,segments(1),initial_condition,input_Power_pump_backward);

% Initialize N2 to be exported, the ion density of the upper state
if gain_rate_eqn.export_N2
    if sim.single_yes
        N2 = zeros([size(N_total) save_points],'single');
    else
        N2 = zeros([size(N_total) save_points]);
    end
end

% Setup a small matrix to track the number of iterations per step
if isequal(sim.step_method,'MPA')
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
        'Name',sprintf('Running GMMNLSE: %s...',sim.progress_bar_name),...
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
        clean_saved_mat(gain_rate_eqn,num_segments);
        
        error('GMMNLSE_propagate:ProgressBarBreak',...
        'The "cancel" button of the progress bar has been clicked.');
    end
    
    % =====================================================================
    % GMMNLLSE: Run the correct step function depending on the options chosen.
    % =====================================================================
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
        last_Power_ASE_backward     = Power_ASE_backward    {zi-1};
        last_signal_fields_backward = signal_fields_backward{zi-1};

        if Zi == 2 % the first/starting z_point
            last_Power_pump_forward  = Power_pump_forward {1};
            last_Power_pump_backward = Power_pump_backward{1};
            last_Power_ASE_forward   = Power_ASE_forward  {1};
            last_signal_fields       = signal_fields{1};
        end
    end

    % For MPA, fundamental mode treats the gain as a dispersion term, 
    %    while multimode treats it as a nonlinear term due to the performance.
    % For split step, only fundamental-mode cases is available because it's hard to calculate the amplification for the gain-dispersion term if the field at a certain time/frequency is too small.
    % I don't implement the gain as a nonlinear term for the split-step algorithm because it just takes too much time to compute for the Runge-Kutta scheme.
    if isequal(sim.step_method,'MPA')
        if isequal(gain_rate_eqn.midx,1) % fundamental mode
            if gain_rate_eqn.export_N2
                [num_it,last_signal_fields,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,N2_next] = GMMNLSE_MPA_step_SMrategain_linear_oscillator(last_signal_fields,last_signal_fields_backward,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,last_Power_ASE_backward,dt,sim,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN);
            else
                [num_it,last_signal_fields,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward]         = GMMNLSE_MPA_step_SMrategain_linear_oscillator(last_signal_fields,last_signal_fields_backward,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,last_Power_ASE_backward,dt,sim,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN);
            end
        else % multimode or higher-order modes
            if gain_rate_eqn.export_N2
                [num_it,last_signal_fields,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,N2_next] = GMMNLSE_MPA_step_MMrategain_linear_oscillator(last_signal_fields,last_signal_fields_backward,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,last_Power_ASE_backward,dt,sim,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN);
            else
                [num_it,last_signal_fields,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward]         = GMMNLSE_MPA_step_MMrategain_linear_oscillator(last_signal_fields,last_signal_fields_backward,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,last_Power_ASE_backward,dt,sim,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN);
            end
        end
        iterations_hist(num_it) = iterations_hist(num_it)+1;
    else % split-step, RK4IP
        if isequal(gain_rate_eqn.midx,1) % fundamental mode
            if gain_rate_eqn.export_N2
                [       last_signal_fields,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,N2_next] = GMMNLSE_ss_step_rategain_linear_oscillator(   last_signal_fields,last_signal_fields_backward,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,last_Power_ASE_backward,dt,sim,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN);
            else
                [       last_signal_fields,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward]         = GMMNLSE_ss_step_rategain_linear_oscillator(   last_signal_fields,last_signal_fields_backward,last_Power_pump_forward,last_Power_pump_backward,last_Power_ASE_forward,last_Power_ASE_backward,dt,sim,prefactor,SRa_info,SRb_info,SK_info,omegas,D_op,haw,hbw,gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN);
            end
        else % multimode or higher-order modes
            error('GMMNLSE_gain_rate_eqn:AlgorithmError',...
                'For multimode cases, there''s only MPA available because there will be some problem with calculating the amplification for the dispersion term for the split-step algorithm.');
        end
    end

    if rem(Zi-1, num_large_steps_persave) == 0
        Power_pump_forward {int64((Zi-1)/num_large_steps_persave+1)} = last_Power_pump_forward;
        Power_pump_backward{int64((Zi-1)/num_large_steps_persave+1)} = last_Power_pump_backward;
    end
    Power_ASE_forward{zi} = last_Power_ASE_forward;
    signal_fields    {zi} = last_signal_fields;

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

    % =====================================================================
    % Others
    % =====================================================================
    % Check for any NaN elements, if desired
    if sim.check_nan
        if any(any(isnan(last_signal_fields))) %any(isnan(last_signal_fields),'all')
            if ~gain_rate_eqn.save_all_in_RAM
                clean_saved_mat(gain_rate_eqn,num_segments);
            end
            
            error('GMMNLSE_propagate:NaNError',...
                'NaN field encountered, aborting.\nPossible reason is that the nonlinear length is too close to the large step size, deltaZ*M for MPA or deltaZ for split-step.');
        end
    end
    
    % Center the pulse
    last_signal_fields_in_time = fft(last_signal_fields);
    tCenter = floor(sum(sum((-floor(N/2):floor((N-1)/2))'.*abs(last_signal_fields_in_time).^2),2)/sum(sum(abs(last_signal_fields_in_time).^2),2));
    if ~isnan(tCenter) && tCenter ~= 0 % all-zero fields; for calculating ASE power only
        % Because circshift is slow on GPU, I discard it.
        %last_signal_fields = ifft(circshift(last_signal_fields_in_time,-tCenter));
        if tCenter > 0
            last_signal_fields = ifft([last_signal_fields_in_time(1+tCenter:end,:);last_signal_fields_in_time(1:tCenter,:)]);
        elseif tCenter < 0
            last_signal_fields = ifft([last_signal_fields_in_time(end+1+tCenter:end,:);last_signal_fields_in_time(1:end+tCenter,:)]);
        end
        if sim.gpu_yes
            t_delay = t_delay + gather(tCenter*dt);
        else
            t_delay = t_delay + tCenter*dt;
        end
    end
    
    % Save and load files
    if save_files
        current_segment_idx = ceil(Zi/num_each_segment);
        next_segment_idx = current_segment_idx + 1;
        if gain_rate_eqn.save_all_in_RAM && sim.gpu_yes
            cumsum_segments = [0 cumsum(segments)];
            starti = cumsum_segments(current_segment_idx)+1;
            endi   = cumsum_segments(current_segment_idx+1);
            [signal_fields,signal_fields_backward,Power_ASE_forward,Power_ASE_backward] = mygather2(starti,endi,signal_fields,signal_fields_backward,Power_ASE_forward,Power_ASE_backward);
        else
            if sim.gpu_yes
                [signal_fields,signal_fields_backward,Power_ASE_forward,Power_ASE_backward] = mygather(signal_fields,signal_fields_backward,Power_ASE_forward,Power_ASE_backward);
            end
            save(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,current_segment_idx),'signal_fields','signal_fields_backward','Power_ASE_forward','Power_ASE_backward','-v7.3');
        end
        
        if next_segment_idx <= length(segments) % if z_points_all=num_each_segments*length(segments) exactly, next_segment_idx can be larger than length(segments). A check is provided here.
            if gain_rate_eqn.save_all_in_RAM && sim.gpu_yes
                starti = cumsum_segments(next_segment_idx)+1;
                endi   = cumsum_segments(next_segment_idx+1);
                [signal_fields_backward,Power_ASE_backward] = mygpuArray2(starti,endi,signal_fields_backward,Power_ASE_backward);
            else
                load(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,next_segment_idx),'signal_fields_backward','Power_ASE_backward');
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
% if num_segments=1, it'll be saved in GMMNLSE_gain_rate_eqn, the caller function of this sub-function;
% if num_segments>1, the latest one will be saved below.
if ~gain_rate_eqn.save_all_in_RAM || ~sim.gpu_yes
    if num_segments ~= 1
        [signal_fields,signal_fields_backward,Power_ASE_forward,Power_ASE_backward] = mygather(signal_fields,signal_fields_backward,Power_ASE_forward,Power_ASE_backward);
        save(sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,num_segments),'signal_fields','signal_fields_backward','Power_ASE_forward','Power_ASE_backward','-v7.3');
        [signal_fields,signal_fields_backward,Power_ASE_forward,Power_ASE_backward] = mygpuArray(signal_fields,signal_fields_backward,Power_ASE_forward,Power_ASE_backward);
    end
end

% Output
if isequal(sim.step_method,'MPA')
    if gain_rate_eqn.export_N2
        N2 = prep_for_output_N2(N2,N_total,num_large_steps_persave);
        varargout = {signal_fields,signal_fields_backward,t_delay,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward,N2,full_iterations_hist};
    else
        varargout = {signal_fields,signal_fields_backward,t_delay,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward,   full_iterations_hist};
    end
else % split-step, RK4IP
    if gain_rate_eqn.export_N2
        N2 = prep_for_output_N2(N2,N_total,num_large_steps_persave);
        varargout = {signal_fields,signal_fields_backward,t_delay,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward,N2};
    else
        varargout = {signal_fields,signal_fields_backward,t_delay,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward};
    end
end

end

%% initialization
function [signal_fields,Power_pump_forward,Power_pump_backward,Power_ASE_forward] = initialization(sim,gain_rate_eqn,first_segment,N,num_modes,save_points,segment1,initial_condition,input_Power_pump_backward)
%INITIALIZATION
%
%   The reason I use cell arrays instead of a matrix (N,num_modes,z_points) for signal_fields and Power:
%       It's faster!
%       e.g. "signal_fields(:,:,zi) = signal_fields_next" is very slow.

    function output = initialize_zeros(mat_size,z_points,single_yes)
        output = cell(1,1,z_points);
        if single_yes
            output(:) = {zeros(mat_size,'single')};
        else
            output(:) = {zeros(mat_size)};
        end
    end

% Signal fields
signal_fields = initialize_zeros([N,num_modes],first_segment,sim.single_yes);
% Pump
Power_pump_forward  = initialize_zeros(1,save_points,sim.single_yes);
Power_pump_backward = initialize_zeros(1,save_points,sim.single_yes);
% ASE
if gain_rate_eqn.ignore_ASE % Because ASE is ignored, set it a scalar zero is enough.
    Power_ASE_forward = initialize_zeros(1,first_segment,sim.single_yes);
else
    Power_ASE_forward = initialize_zeros([1,1,num_modes,1,N],first_segment,sim.single_yes); % Make it the size to (1,1,num_modes,1,N) for "solve_gain_rate_eqn.m"
end

% Put in the necessary information.
% Signal fields
if sim.single_yes
    signal_fields{1} = single(initial_condition.fields);
else
    signal_fields{1} = initial_condition.fields;
end
% Pump power
if any(strcmp(gain_rate_eqn.pump_direction,{'co','bi'}))
    if sim.single_yes
        Power_pump_forward{1} = single(gain_rate_eqn.copump_power);
    else
        Power_pump_forward{1} = gain_rate_eqn.copump_power;
    end
end
if any(strcmp(gain_rate_eqn.pump_direction,{'counter','bi'}))
    if sim.single_yes
        Power_pump_backward{1} = single(input_Power_pump_backward);
    else
        Power_pump_backward{1} = input_Power_pump_backward;
    end
end
% ASE power
if ~gain_rate_eqn.ignore_ASE
    if sim.single_yes
        Power_ASE_forward{1} = single(permute(initial_condition.ASE_forward,[3 4 2 5 1]));
    else
        Power_ASE_forward{1} = permute(initial_condition.ASE_forward,[3 4 2 5 1]);
    end
end

% -------------------------------------------------------------------------
% GPU
if sim.gpu_yes
    if gain_rate_eqn.save_all_in_RAM
        [Power_pump_forward,Power_pump_backward] = mygpuArray(Power_pump_forward,Power_pump_backward);
        [signal_fields,Power_ASE_forward] = mygpuArray2(1,segment1,signal_fields,Power_ASE_forward);
    else
        [signal_fields,Power_pump_forward,Power_pump_backward,Power_ASE_forward] = mygpuArray(signal_fields,Power_pump_forward,Power_pump_backward,Power_ASE_forward);
    end
end

end

%% EXTRACT_CELL_CONTENT
function x = extract_cell_content(x)
    if iscell(x)
        x = x{1};
    end
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

%% CLEAN_SAVED_MAT
function clean_saved_mat(gain_rate_eqn,num_segments)
% CLEAN_SAVED_MAT It deletes the saved mat files for iterations.

for segment_idx = 1:num_segments
    current_caller_path = cd;
    if (exist(fullfile(current_caller_path, sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,segment_idx)), 'file') == 2)
        delete( fullfile(current_caller_path, sprintf('%s%u.mat',gain_rate_eqn.saved_mat_filename,segment_idx)) );
    end
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