function [ gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN ] = gain_info( sim,gain_rate_eqn,lambda,mode_profiles )
%GAIN_INFO Computes several information related to the gain medium.
%
% =========================================================================
% =============== Call this function with the following code ==============
% =========================================================================
% f = ifftshift( (-N/2:N/2-1)/N/dt + sim.f0 ); % in the order of "omegas" in the "GMMNLSE_propagate.m"
% c = 299792.458; % nm/ps
% lambda = c./f; % nm
%
% % For single mode,
% [gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total] = gain_info( sim,gain_rate_eqn,lambda );
%
% % For multimode,
% [gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN] = gain_info( sim,gain_rate_eqn,lambda );
%
% And then send these variables all in "GMMNLSE_propagate.m" with a cell:
%
%   % For single mode,
%   gain_param = {gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total};
%   output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);
%
%   % For multimode,
%   gain_param = {gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN};
%   output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);
% =========================================================================
%
%   gain_rate_eqn:
%
%       multimode mode-profile folder -->
%
%           MM_folder - a string; where the betas.mat and S_tensor_?modes.mat are
%
%       oscillator info -->
%
%           reuse_data - true(1) or false(0);
%                        For a ring or linear cavity, the pulse will enter a steady state eventually.
%                        If reusing the pump and ASE data from the previous roundtrip, the convergence can be much faster, especially for counterpumping.
%
%           linear_oscillator_model - 0: not a linear oscillator
%                                     1: a linear oscillator (faster algorithm)
%                                     2: a linear oscillator
%                               For a linear oscillator, there are pulses from both directions simultaneously, which will both contribute to saturating the gain;
%                               therefore , the backward-propagating pulses need to be taken into account.
%
%                               How to use it:
%                                   prop_output = GMMNLSE_propagate(fiber, input_field, sim, rate_gain_saved_data);
%                                   (next)rate_gain_saved_data = prop_output.saved_data;
%
%                                   "rate_gain_saved_data" contains
%                                   	signal_fields, signal_fields_backward,
%                                   	Power_pump_forward, Power_pump_backward,
%                                       Power_ASE_forward, Power_ASE_backward
%
%           save_all_in_RAM - true(1) or false(0);
%                             Save all the required info in the RAM instead of MAT files with the filename defined below
%           saved_mat_filename - a string; the files used to temporarily store the data during iterations if necessary
%                                (default: GMMNLSE_tmp)
%
%       fiber info -->
%
%           core_diameter - um; where the doped ion stays
%           cladding_diameter - um
%           core_NA - numerical aperture of the gain fiber
%
%       doped ion info -->
%
%           absorption_wavelength_to_get_N_total - nm
%           absorption_to_get_N_total - dB/m
%           cross_section_filename - the filename for the doped ion cross section data
%                                    Currently I have 'Liekki Yb_AV_20160530.txt' for Yb and 'optiwave Er' for Er
%                                    Er data is from optiwave website: https://optiwave.com/resources/applications-resources/optical-system-edfa-basic-concepts/
%
%       pump info -->
%
%           pump_wavelength - nm
%           pump_direction - "Don't set this parameter" since it'll be automatically determined from "copump_power" and "counterpump_power" below.
%                            'co', 'counter', or 'bi'
%                            'co' reads only the copump_power, 'counter' reads only the counterpump_power, while 'bi' reads both.
%           copump_power - W
%           counterpump_power - W
%       
%       modes inside the gain core area (if not specified, it'll be loaded in this code and gain_rate_eqn will be upated) -->
%
%           midx - (1,num_modes) array; the mode index
%           mode_volume - the total number of spatial modes available in the fiber
%           downsampling_factor - reduce the size of mode profiles for multimode
%
%       computational info -->
%
%           tau - the lifetime of the upper states, which is used in spontaneous emission; in "s".
%                 (1) lifetime of Yb in F_(5/2) state = 840us (Paschotta et al., "Lifetme quenching in Yb-doped fibers")
%                 (2) lifetime of Er in (^4)I_(13/2) state = 8-10ms (Z. Y. Zhang et al., "Fluorescence decay-time characteristics of erbium-doped optical fiber at elevated temperatures")
%           t_rep - the roundtrip time (1/repetition rate) of the pulse, which is used to calculate the power of the "signal" pulse; in "s"
%
%       rate equation model algorithm info -->
%
%           export_N2 - 1(true) or 0(false); Whether to export N2, the ion density of the upper state, or not
%           ignore_ASE - 1(true) or 0(false)
%           iterate - 1(true) or 0(false); The coupled equations of the rate equations needs to be computed back and forth to get the converged result.
%           max_iterations - the maximum number of iterations
%           tol - the tolerance of this iteration loop. If the difference between the last two results is smaller than this tolerance, we're done. 
%           allow_coupling_to_zero_fields - 1(true) or 0(false); 
%                                           If there are all-zero fields, there will be coupling into those fields due to the rate equations.
%                                           Because it's so weak for each small z step, the value fluctuates relatively large and makes MPA large to converge.
%                                           "double-precision" is necessary for this to run. Even so, it's still found to be really hard to converge for MPA because of the amplification of those all-zero fields.
%           verbose - show the information(final pulse energy) during iterations
%           gpuDevice - the output of MATLAB function, "gpuDevice" 
%                       (This can be found by default.)
%           memory_limit: the RAM limit for this computation
%                         (This can be found by default.)
%
%   lambda - the wavelengths of the computational region; in "nm"
%
%   mode_profiles (If not specified, this code will read them from "gain_rate_eqn.MM_folder"):
%
%       mode_profiles - (Nx,Nx,num_spatial_modes); the eigenmode field profiles of modes;
%                       It'll be normalized into the unit of "1/um"
%       mode_profiles_x - the x-position for mode_profiles; in "um"
%
%   =======================================================================
%   Output:
%
%       cross_sections_pump - um^2
%       cross_sections - um^2
%       overlap_factor - no unit for single-mode and 1/um^2 for multimode
%       N_total - (Nx,Nx); the doped ion density; in "1/um^3"
%       FmFnN - precalculate the integral2(overlap_factor*N_total) for the signal and ASE
%       GammaN - precalculate the integral2(overlap_factor*N_total) for the pump

%% Add the folder of functions of gain-rate-equation model and its functions
% Besides loading mode coupling folder, this "sep_char" is also used in GPU setup below.
if ispc
    sep_char = '\';
else % unix
    sep_char = '/';
end
current_path = mfilename('fullpath');
sep_pos = strfind(current_path,sep_char);
current_folder = current_path(1:sep_pos(end));
addpath([current_folder 'Gain_rate_eqn/']);

%% Specify the mat filename of the temporarily saved data during propagation
% If this calculation requires iterations, all the data along the
% propagation needs to be saved and updated during iterations. Since
% all data at all small steps are saved, dz, this contains lots of data.
% Instead of keeping them all in the memory, I save the file out.
if ~isfield(gain_rate_eqn,'saved_mat_filename') || isempty(gain_rate_eqn.saved_mat_filename)
    gain_rate_eqn.saved_mat_filename = 'GMMNLSE_tmp';
end

%% linear oscillator model
switch gain_rate_eqn.linear_oscillator_model
    case 0 % not a linear oscillator
        gain_rate_eqn.linear_oscillator = false;
    case 1
        gain_rate_eqn.linear_oscillator = true;
        
        if ~gain_rate_eqn.ignore_ASE
            error('gain_info:LinearOscillatorModelError',...
                  'If considering ASE, use linear oscillator model 2. Model 1 will fail because updating initial condition of ASE for each pass of the fiber isn''t enough and it''s been tested.');
        end
    case 2
        gain_rate_eqn.linear_oscillator = true;
        addpath([current_folder 'Gain_rate_eqn/linear_oscillator/']);
end
if gain_rate_eqn.linear_oscillator
    gain_rate_eqn.reuse_data = true; % Force to reuse the previous calculated data because it's a linear oscillator
end

%% Yb Cross sections
% "lambda" must be a column vector.
if size(lambda,1) == 1
    lambda = lambda.';
end
if issorted(lambda,'monotonic')
    lambda = ifftshift(lambda,1);
end
% Read cross sections from the file.
necessary_lambda = [gain_rate_eqn.absorption_wavelength_to_get_N_total*1e-9; ...
                    gain_rate_eqn.pump_wavelength*1e-9; ...
                    lambda*1e-9];
[absorption,emission] = read_cross_sections(gain_rate_eqn.cross_section_filename,necessary_lambda); % read the file
absorption = absorption*1e12; % change the unit to um^2
emission = emission*1e12;

cross_sections_pump = struct('absorption',absorption(2),'emission',emission(2)); % pump
cross_sections = struct('absorption',absorption(3:end),'emission',emission(3:end)); % signal, ASE
cross_sections = structfun(@(x) permute(x,[2 3 4 5 1]),cross_sections,'UniformOutput',false); % change it to the size (1,1,1,1,N)

%% Overlap factor of the field and the dopping area
% Load mode profiles for multimode
if ~isequal(gain_rate_eqn.midx,1) % multimode or higher-order modes
    if ~exist('mode_profiles','var')
        lambda0 = int16(lambda(1));
        mode_solver_field = {'EX','EY','scalar'};
        for i = 1:length(mode_solver_field)
            if exist(sprintf('%sradius%uboundary0000field%smode%uwavelength%u.mat',gain_rate_eqn.MM_folder,round(gain_rate_eqn.core_diameter/2),mode_solver_field{i},gain_rate_eqn.midx(1),lambda0),'file')
                break;
            end
        end
        mode_solver_field = mode_solver_field{i};
        % load(sprintf('%sradius%uboundary0000field%smode%03.fwavelength%u.mat',gain_rate_eqn.MM_folder,round(gain_rate_eqn.core_diameter/2),mode_solver_field,gain_rate_eqn.midx(1),lambda0),'phi','x'); % load 1st mode first to get the size and dimension of the mode profile
        % load 1st mode first to get the size and dimension of the mode profile

        % added by BarakM 11.5.24
        fname=[gain_rate_eqn.MM_folder 'radius' strrep(num2str(gain_rate_eqn.core_diameter/2), '.', '_') 'boundary0000' 'field' mode_solver_field  'mode' num2str(gain_rate_eqn.midx(1),'%03.f') 'wavelength' strrep(num2str(lambda0), '.', '_')];
        load([fname '.mat'], 'phi','x');


        mode_profiles.mode_profile_x = x; % x-position vector
        mode_profiles.mode_profiles = zeros(length(x),length(x),length(gain_rate_eqn.midx)); % initialization
        mode_profiles.mode_profiles(:,:,1) = phi; % the 1st mode
        for ni = 2:length(gain_rate_eqn.midx)
            n = gain_rate_eqn.midx(ni);
            % load(sprintf('%sradius%uboundary0000field%smode%03.fwavelength%u.mat',gain_rate_eqn.MM_folder,round(gain_rate_eqn.core_diameter/2),mode_solver_field,n,lambda0),'phi');
            
            % added by BarakM 11.5.24
            fname=[gain_rate_eqn.MM_folder 'radius' strrep(num2str(gain_rate_eqn.core_diameter/2), '.', '_') 'boundary0000' 'field' mode_solver_field  'mode' num2str(n,'%03.f') 'wavelength' strrep(num2str(lambda0), '.', '_')];
            load([fname '.mat'], 'phi');
            
            mode_profiles.mode_profiles(:,:,ni) = phi;
        end
    end
    gain_rate_eqn.mode_profile_dx = abs(mode_profiles.mode_profile_x(2)-mode_profiles.mode_profile_x(1)); % unit: um; This variable will be used later in "solve_gain_rate_eqn", so I put it in "gain_rate_eqn".
    % Normalization
    norms = sqrt(sum(sum( abs(mode_profiles.mode_profiles).^2 ,1),2))*gain_rate_eqn.mode_profile_dx;
    mode_profiles.mode_profiles = mode_profiles.mode_profiles./norms; % unit: 1/um
    
    % Choose only the core region for the mode profiles to save the memory.
    % Also downsample the spatial profiles.
    chosen_region_left_idx = find(mode_profiles.mode_profile_x > -gain_rate_eqn.core_diameter/2,1) - 1;
    chosen_region_right_idx = find(mode_profiles.mode_profile_x < gain_rate_eqn.core_diameter/2,1,'last') + 1;
    core_region_idx = chosen_region_left_idx:chosen_region_right_idx;
    mode_profiles.mode_profile_x = downsample(mode_profiles.mode_profile_x(core_region_idx),gain_rate_eqn.downsampling_factor);
    old_large_mode_profiles = mode_profiles.mode_profiles;
    mode_profiles.mode_profiles = zeros(length(mode_profiles.mode_profile_x),length(mode_profiles.mode_profile_x),length(gain_rate_eqn.midx));
    for ni = 1:length(gain_rate_eqn.midx)
        mode_profiles.mode_profiles(:,:,ni) = downsample(downsample(old_large_mode_profiles(core_region_idx,core_region_idx,ni), gain_rate_eqn.downsampling_factor)', gain_rate_eqn.downsampling_factor)'; % downsample;;
    end
    
    gain_rate_eqn.mode_profile_dx = gain_rate_eqn.mode_profile_dx*gain_rate_eqn.downsampling_factor;
    
    % Core region where active ions lie
    [x,y] = meshgrid(mode_profiles.mode_profile_x,mode_profiles.mode_profile_x);
    core_region = (x.^2 + y.^2) <= (gain_rate_eqn.core_diameter/2)^2;
    
    % Consider the core region only
    mode_profiles.mode_profiles = mode_profiles.mode_profiles.*core_region;
end

% pump overlap factor = A_doping/A_cladding
if isequal(gain_rate_eqn.midx,1) % fundamental-mode
    overlap_factor.pump = 1/(pi*(gain_rate_eqn.cladding_diameter/2)^2);
else % multimode
    overlap_factor.pump = 1/(pi*(gain_rate_eqn.cladding_diameter/2)^2)*core_region;
end

% signal overlap factor = F_m*conj(F_n)
if isequal(gain_rate_eqn.midx,1) % fundamental-mode
    V = 2*pi*(gain_rate_eqn.core_diameter/2)/(lambda(1)*1e-3)*gain_rate_eqn.core_NA; % V-number, normalized frequency
    if V < 0.8 || V > 2.8
        warning('For the computation of the fundamental mode, I use "Whitley''s Gaussian-mode approximation. It works only in the range of V of 0.8-2.8.');
    end
    % w_over_a = 0.65+1.619*V^(-3/2)+2.879*V^(-6); % Marcuse et al., "Loss analysis of single-mode fiber splices" (1977)
    w_over_a = 0.616+1.66*V^(-3/2)+0.987*V^(-6); % Whitley et al., "Alternative Gaussian spot size polynomial for use with doped fiber amplifiers" (1993)
    overlap_factor.signal = ( 1-exp(-2/w_over_a^2) )/(pi*(gain_rate_eqn.core_diameter/2)^2);
else % multimode or higher-order modes
    overlap_factor.signal = mode_profiles.mode_profiles.*permute(conj(mode_profiles.mode_profiles),[1 2 4 3]);
end

%% Doped ion density
% For small-signal absorption, N2~0 and N1~N_total.
% pump is proportional to "exp(-integral2(overlap_factor.pump)*N_total*absorption_cross_section*L)", L: propagation length
% absorption_dB/m =10*log10( exp(-integral2(overlap_factor.pump)*N_total*absorption_cross_section*(L=1m)) )
N_total = log(10^(gain_rate_eqn.absorption_to_get_N_total/10))./(((gain_rate_eqn.core_diameter/gain_rate_eqn.cladding_diameter)^2)*absorption(1)*1e6); % doped ion density based on absorption at a specific wavelength; in "1/um^3"
if ~isequal(gain_rate_eqn.midx,1) % multimode or higher-order modes
    N_total = N_total*core_region; % size: (Nx,Nx)
end

%% Mode volume
% Spontaneous emission emits photons in all the possible modes with an 
% equal probability, so we need to know the total number of modes that a
% fiber has.
% A parabolic-index fiber is assumed here, from 
% Mafi et.al, Pulse Propagation in a Short Nonlinear Graded-Index Multimode Optical Fiber (2012)
if ~gain_rate_eqn.ignore_ASE
    if isempty(gain_rate_eqn.mode_volume)
        % Calculate the index difference using the Sellmeier equation to generate n(lambda)
        a1=0.6961663;
        a2=0.4079426;
        a3=0.8974794;
        b1= 0.0684043;
        b2=0.1162414;
        b3=9.896161;

        n_from_Sellmeier = @(lambda) (1+a1*(lambda.^2)./(lambda.^2 - b1^2)+a2*(lambda.^2)./(lambda.^2 - b2^2)+a3*(lambda.^2)./(lambda.^2 - b3^2)).^(0.5);
        n0 = n_from_Sellmeier(lambda(1)*1e-3);
        n_clad = sqrt(n0^2-gain_rate_eqn.core_NA^2);
        
        gain_rate_eqn.mode_volume = ceil((n0*pi/lambda(1)*1e3*gain_rate_eqn.core_diameter/2)^2*(n0-n_clad)/n0);
    end
end

%% Check the validity of the code
% Because I use the approximation, sqrt(1+x)=1+x/2 if x is small, in
% calculating signal fields for multimode, the code will give error here if
% this approximation is bad.
if sim.adaptive_deltaZ.model ~= 0 && ... % not using adaptive-deltaZ method
    ~isequal(gain_rate_eqn.midx,1) % multimode or higher-order modes
    tol_approximation = 1e-3;
    approx_error = @(x)abs((sqrt(1+x)-(1+x/2))./sqrt(1+x));
    if approx_error( (sim.deltaZ/2*1e6)*max(N_total(:))*max(cross_sections.absorption) ) > tol_approximation
        error('gain_info:deltaZError',...
            'The deltaZ is too large for this code to run because of the approximation of "sqrt(1+x)=1+x/2" I use for calculating the gain for multimode cases.');
    end
end

% This code assumes the population inversion reaches the steady state
% because of high repetition rate of pulses.
% I'll check the highest recovery lifetime for the inversion. This should
% be larger than repetition rate for this assumption to be true.
h = 6.626e-34;
c = 299792458;
tc = 1/(max(overlap_factor.pump(:))*(gain_rate_eqn.pump_wavelength*1e-9)/(h*c)*(cross_sections_pump.absorption+cross_sections_pump.emission)*max(gain_rate_eqn.copump_power,gain_rate_eqn.counterpump_power)+1/gain_rate_eqn.tau);
if tc < gain_rate_eqn.t_rep
    warning('The repetition rate isn''t high enough for this code to be accurate.');
end

%% Pre-compute the integral of "overlap_factor*N_total"
if ~isequal(gain_rate_eqn.midx,1) % multimode or higher-order modes
    trapz2 = @(x) trapz(trapz(x,1),2)*gain_rate_eqn.mode_profile_dx^2; % take the integral w.r.t. the x-y plane
    FmFnN = trapz2(overlap_factor.signal.*N_total);
    GammaN = trapz2(overlap_factor.pump.*N_total);
else
    FmFnN = [];
    GammaN = [];
end

%% Change the data type
if sim.single_yes
    cross_sections_pump = structfun(@(c) single(c),cross_sections_pump,'UniformOutput',false);
    cross_sections = structfun(@(c) single(c),cross_sections,'UniformOutput',false);
    overlap_factor = structfun(@(c) single(c),overlap_factor,'UniformOutput',false);
    N_total = single(N_total);
    FmFnN = single(FmFnN);
    GammaN = single(GammaN);
end

%% Query the gpuDevice
if sim.gpu_yes
    sim.gpuDevice.Device = gpuDevice(sim.gpuDevice.Index); % use the specified GPU device
end

%% Find the memory limit
if ~isfield(gain_rate_eqn,'memory_limit')
    if sim.gpu_yes
        gain_rate_eqn.memory_limit = sim.gpuDevice.Device.AvailableMemory/2;
    else
        if ispc % Windows
            userview = memory;
            gain_rate_eqn.memory_limit = userview.MemUsedMATLAB/2*2^20; % B
        elseif isunix % Unix, linux
            [~,w] = unix('free -b | grep Mem'); % Display the memory info in Bytes
            stats = str2double(regexp(w, '[0-9]*', 'match'));
            %memsize = stats(1)/1e6;
            freemem = stats(end); % B; availabel memory
            gain_rate_eqn.memory_limit = freemem/2;
        else % iOS
            error('OSError:GMMNLSE_gain_rate_eqn',...
                  'The "save_all_in_RAM" function doesn''t support iOS.');
        end
    end
end

%% Save info
if ~isfield(gain_rate_eqn,'save_all_in_RAM')
    % Because typically GPU has less memory than RAM, I don't put
    % everything in memory once.
    if sim.gpu_yes || ~gain_rate_eqn.reuse_data
        gain_rate_eqn.save_all_in_RAM = false;
    else
        gain_rate_eqn.save_all_in_RAM = true;
    end
end

end