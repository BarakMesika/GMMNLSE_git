function varargout = GMMNLSE_adaptive_rategain(sim,gain_rate_eqn,...
                                               cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,...
                                               save_z,save_deltaZ,save_points,t_delay_out,...
                                               initial_condition,...
                                               prefactor, omegas, D_op,...
                                               SK_info, SRa_info, SRb_info, haw, hbw)
%GMMNLSE_ADAPTIVE_RATEGAIN It attains the field after propagation inside the gain 
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
%       The adaptive scheme works only for forward propagation without ASE
%       or reusing the data (for an oscillator to converge faster).
%
%       If there's backward-propagating fields, that is, ASE or
%       counterpumping, iterations are necessary to get a more accurate
%       result. For these cases, please use the non-adaptive scheme.

%% Error check
if gain_rate_eqn.reuse_data || gain_rate_eqn.linear_oscillator_model ~= 0 || gain_rate_eqn.save_all_in_RAM
    error('GMMNLSE_adaptive_rategain:settingsError',...
          'Adaptive-step method doesn''t support reuse_data, linear_oscillator_model, and save_all_in_RAM.');
end

%%
N = size(initial_condition.fields,1);

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
if ~doesnt_need_iterations
    error('GMMNLSE_adaptive_rategain:propagationDirectionError',...
          'Adaptive scheme works only for forward propagation/co-pumping.')
end
if gain_rate_eqn.reuse_data
    error('GMMNLSE_adaptive_rategain:propagationDirectionError',...
          'Adaptive scheme works only for forward propagation/co-pumping without reusing the data (for an oscillator to converge faster).')
end

%% Propagations
% Start the pulse propagation.
if gain_rate_eqn.export_N2
    [signal_fields,Power_pump_forward,...
     save_i,save_z,save_deltaZ,...
     t_delay_out,...
     N2]          = gain_propagate(sim,gain_rate_eqn,...
                                   save_points,save_z,save_deltaZ,t_delay_out,...
                                   N,prefactor,omegas,D_op,...
                                   SK_info,SRa_info,SRb_info,haw,hbw,...
                                   cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,...
                                   initial_condition);
else
    [signal_fields,Power_pump_forward,...
     save_i,save_z,save_deltaZ,...
     t_delay_out] = gain_propagate(sim,gain_rate_eqn,...
                                   save_points,save_z,save_deltaZ,t_delay_out,...
                                   N,prefactor,omegas,D_op,...
                                   SK_info,SRa_info,SRb_info,haw,hbw,...
                                   cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,...
                                   initial_condition);
end

% Output:
% Power_out is in frequency domain while field_out is transformed into time domain.

% Transform the current "signal_fields" and "Power..." into arrays.
if sim.gpu_yes
    [signal_fields,Power_pump_forward] = mygather(signal_fields,Power_pump_forward);
    [signal_fields_out,Power_pump_forward_out] = deal(signal_fields,Power_pump_forward);
else
    [signal_fields_out,Power_pump_forward_out] = deal(signal_fields,Power_pump_forward);
end
signal_fields_out      = cell2mat(signal_fields_out);
Power_pump_forward_out = cell2mat(Power_pump_forward_out);

Power_out = struct('pump_forward',Power_pump_forward_out);
signal_fields_out = fft(signal_fields_out);

if gain_rate_eqn.export_N2
    varargout = {signal_fields_out,Power_out,...
                 save_i,save_z,save_deltaZ,...
                 t_delay_out,...
                 N2};
else
    varargout = {signal_fields_out,Power_out,...
                 save_i,save_z,save_deltaZ,...
                 t_delay_out};
end

end

%%
function varargout = gain_propagate(sim,gain_rate_eqn,...
                                    save_points,save_z,save_deltaZ,t_delay_out,...
                                    N,prefactor,omegas,D_op,...
                                    SK_info,SRa_info,SRb_info,haw,hbw,...
                                    cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN, ...
                                    initial_condition)
%GAIN_PROPAGATE Runs the corresponding propagation method based on "direction".

if sim.gpu_yes
    dummy_var = zeros(size(initial_condition.fields),'gpuArray');
else
    dummy_var = zeros(size(initial_condition.fields));
end

dt = initial_condition.dt;
num_modes = size(initial_condition.fields,2);

[signal_fields,Power_pump_forward] = initialization(sim,gain_rate_eqn,N,num_modes,save_points,initial_condition);
last_Power_pump_forward = Power_pump_forward{1};
last_signal_fields      = signal_fields     {1}; % = initial_condition.fields

% Initialize N2 to be exported, the ion density of the upper state
if gain_rate_eqn.export_N2
    if sim.single_yes
        N2 = zeros([size(N_total) save_points],'single');
    else
        N2 = zeros([size(N_total) save_points]);
    end
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
    num_progress_updates = 1000;
    progress_bar_z = (1:num_progress_updates)*save_z(end)/num_progress_updates;
    progress_bar_i = 1;
end

% max deltaZ
if ~isfield(sim,'max_deltaZ')
    sim.max_deltaZ = sim.save_period/10;
end

% Because I use the approximation, sqrt(1+x)=1+x/2 if x is small, in
% calculating signal fields for multimode, the code will give error here if
% this approximation is bad.
if ~isequal(gain_rate_eqn.midx,1) % multimode or higher-order modes
    tol_approximation = 1e-3;
    approx_error = @(x)abs((sqrt(1+x)-(1+x/2))./sqrt(1+x));
    while approx_error( (sim.max_deltaZ/2*1e6)*max(N_total(:))*max(cross_sections.absorption) ) > tol_approximation
        sim.max_deltaZ = sim.max_deltaZ/2;
    end
end
sim.deltaZ = sim.max_deltaZ;

% Then start the propagation
z = 0;
t_delay = 0; % time delay
save_i = 2; % the 1st one is the initial field
a5 = [];
sim.last_deltaZ = 1;
while z+eps(z) < save_z(end) % eps(z) here is necessary due to the numerical error
    % Check for Cancel button press
    if sim.progress_bar && getappdata(h_progress_bar,'canceling')
        error('GMMNLSE_propagate:ProgressBarBreak',...
        'The "cancel" button of the progress bar has been clicked.');
    end
    
    ever_fail = false;
    previous_signal_fields = last_signal_fields;
    previous_a5 = a5;

    success = false;
    while ~success
        if ever_fail
            last_signal_fields = previous_signal_fields;
            a5 = previous_a5;
        end

        if gain_rate_eqn.export_N2
            [last_signal_fields, a5,...
             last_Power_pump_forward,...
             opt_deltaZ,success,...
             N2_next]            = GMMNLSE_RK4IP_rategain_adaptive(last_signal_fields,dt,sim,gain_rate_eqn,...
                                                                   SK_info,SRa_info,SRb_info,...
                                                                   haw,hbw,...
                                                                   prefactor,omegas,D_op,...
                                                                   cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,...
                                                                   last_Power_pump_forward,a5,...
                                                                   dummy_var);
        else
            [last_signal_fields, a5,...
             last_Power_pump_forward,...
             opt_deltaZ,success] = GMMNLSE_RK4IP_rategain_adaptive(last_signal_fields,dt,sim,gain_rate_eqn,...
                                                                   SK_info,SRa_info,SRb_info,...
                                                                   haw,hbw,...
                                                                   prefactor,omegas,D_op,...
                                                                   cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,...
                                                                   last_Power_pump_forward,a5,...
                                                                   dummy_var);
        end
        
        if ~success
            ever_fail = true;

            sim.deltaZ = opt_deltaZ;
        end
        
        % =====================================================================
        % Others
        % =====================================================================
        % Check for any NaN elements, if desired
        if sim.check_nan
            if any(any(isnan(last_signal_fields))) %any(isnan(last_signal_fields),'all')
                error('GMMNLSE_propagate:NaNError',...
                    'NaN field encountered, aborting.\nPossible reason is that the nonlinear length is too close to the large step size, deltaZ*M for MPA or deltaZ for split-step.');
            end
        end
    end
    sim.last_deltaZ = sim.deltaZ; % previous deltaZ

    % Center the pulse
    last_signal_fields_in_time = fft(last_signal_fields);
    tCenter = floor(sum(sum((-floor(N/2):floor((N-1)/2))'.*abs(last_signal_fields_in_time).^2),2)/sum(sum(abs(last_signal_fields_in_time).^2),2));
    if ~isnan(tCenter) && tCenter ~= 0 % all-zero fields; for calculating ASE power only
        % Because circshift is slow on GPU, I discard it.
        %last_signal_fields = ifft(circshift(last_signal_fields_in_time,-tCenter));
        a5 = fft(a5);
        if tCenter > 0
            a5 = ifft([a5(1+tCenter:end,:);a5(1:tCenter,:)]);
            last_signal_fields = ifft([last_signal_fields_in_time(1+tCenter:end,:);last_signal_fields_in_time(1:tCenter,:)]);
        elseif tCenter < 0
            a5 = ifft([a5(end+1+tCenter:end,:);a5(1:end+tCenter,:)]);
            last_signal_fields = ifft([last_signal_fields_in_time(end+1+tCenter:end,:);last_signal_fields_in_time(1:end+tCenter,:)]);
        end
        if sim.gpu_yes
            tCenter = gather(tCenter);
        end
        t_delay = t_delay + tCenter*initial_condition.dt;
    end
    
    % Update z
    z = z + sim.deltaZ;
    sim.deltaZ = min([opt_deltaZ,save_z(end)-z,sim.max_deltaZ]);
    
    % If it's time to save, get the result from the GPU if necessary,
    % transform to the time domain, and save it
    if gain_rate_eqn.export_N2 && z == sim.last_deltaZ
        
        if sim.gpu_yes
            z_1st_N2 = gather(sim.deltaZ);
            N2(:,:,1) = gather(N2_next(:,:,1,1,1));
        else
            z_1st_N2 = sim.deltaZ;
            N2(:,:,1) = N2_next(:,:,1,1,1);
        end
    end
    if z >= save_z(save_i)
        if sim.gpu_yes
            save_z(save_i) = gather(z);
            save_deltaZ(save_i-1) = gather(sim.last_deltaZ);
            Power_pump_forward{save_i} = gather(last_Power_pump_forward);
            signal_fields{save_i} = gather(last_signal_fields);
            
            if gain_rate_eqn.export_N2
                N2(:,:,save_i) = gather(N2_next(:,:,1,1,1));
            end
        else
            save_z(save_i) = z;
            save_deltaZ(save_i-1) = sim.last_deltaZ;
            Power_pump_forward{save_i} = last_Power_pump_forward;
            signal_fields{save_i} = last_signal_fields;
            
            if gain_rate_eqn.export_N2
                N2(:,:,save_i) = N2_next(:,:,1,1,1);
            end
        end

        t_delay_out(save_i) = t_delay;

        save_i = save_i + 1;
    end
    
    % Report current status in the progress bar's message field
    if sim.progress_bar
        if z >= progress_bar_z(progress_bar_i)
            waitbar(gather(z/save_z(end)),h_progress_bar,sprintf('%s%6.1f%%',sim.progress_bar_name,z/save_z(end)*100));
            progress_bar_i = find(z<progress_bar_z,1);
        end
    end
end

% Output
if gain_rate_eqn.export_N2
    N2 = prep_for_output_N2(N2,N_total,z_1st_N2,save_z);
    varargout = {signal_fields,Power_pump_forward,...
                 save_i,save_z,save_deltaZ,...
                 t_delay_out,...
                 N2};
else
    varargout = {signal_fields,Power_pump_forward,...
                 save_i,save_z,save_deltaZ,...
                 t_delay_out};
end

end

%% initialization
function [signal_fields,Power_pump_forward] = initialization(sim,gain_rate_eqn,N,num_modes,save_points,initial_condition)
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
        output = cell(1,1,save_points);
        if single_yes
            output(:) = {zeros(mat_size,'single')};
        else
            output(:) = {zeros(mat_size)};
        end
    end

% Pump power
Power_pump_forward = initialize_zeros(1,sim.single_yes);
% Put in the necessary information.
if sim.single_yes
    Power_pump_forward{1} = single(gain_rate_eqn.copump_power);
else
    Power_pump_forward{1} = gain_rate_eqn.copump_power;
end

% -------------------------------------------------------------------------
% Signal field
% "cell2mat" doesn't support "gpuArray" in a cell array, which affects the process when getting the output matrix. 
signal_fields = initialize_zeros([N,num_modes],sim.single_yes);
if sim.single_yes
    signal_fields{1} = single(initial_condition.fields);
else
    signal_fields{1} = initial_condition.fields;
end

% -------------------------------------------------------------------------
% GPU
if sim.gpu_yes
    [signal_fields,Power_pump_forward] = mygpuArray(signal_fields,Power_pump_forward);
end

end

%% PREP_FOR_OUTPUT_N2
function N2 = prep_for_output_N2( N2, N_total, z_1st_N2, save_z )
%PREP_FOR_OUTPUT_N2 It interpolates the first z-plane N2 from the other N2
%data and transform N2 into the ratio, N2/N_total.

if isequal(class(N_total),'gpuArray')
    N_total = gather(N_total);
end

N2 = N2/max(N_total(:));
sN2 = size(N2,3)-1;

if sN2 > 1
    N2(:,:,1) = permute(interp1([z_1st_N2;save_z(2:end)],permute(N2,[3 1 2]),0,'spline'),[2 3 1]);
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

%% CLEANMEUP
% -------------------------------------------------------------------------
function cleanMeUp(h_progress_bar)
%CLEANMEUP It deletes the progress bar.

% DELETE the progress bar; don't try to CLOSE it.
delete(h_progress_bar);
    
end
% -------------------------------------------------------------------------