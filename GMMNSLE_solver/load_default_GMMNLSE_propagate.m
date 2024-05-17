function [fiber,sim] = load_default_GMMNLSE_propagate( input_fiber,input_sim,type_of_mode )
%LOAD_DEFAULT_GMMNLSE_PROPAGATE It loads the default settings for "fiber"
%and "sim" for different types of modes used.
%
%   If a user has specified some of the parameters of "fiber" and "sim",
%   user-defined one will be chosen instead of default ones.
%
%   If you want to use a different one, specify it as fiber.xxx or sim.xxx 
%   and send them into this function. The output will use your parameters 
%   besides other default parameters a user doesn't specify.
%
%% ========================================================================
%   Because some parameters are correlated to each other, sometimes it's 
%   necessary to load input parameters first.
%   However, this makes loading default parameters a complicated function.
%
%   The procedure of loading default parameters is described below.
%   The order matters.
% -------------------------------------------------------------------------
%
%   <-- Uncorrelated parameters are loaded directly -->
%
%       sim.f0 - depend on input f0 or lambda0
%                If not input f0 or lambda0, f0=3e5/1030e-9 (THz)
%
%       sim.sw = true
%       sim.deltaZ = 250e-6
%       sim.save_period = 0
%       sim.ellipticity = 0; % linear polarization
%
%       sim.MPA.M = 10
%       sim.MPA.n_tot_max = 20
%       sim.MPA.n_tot_min = 2
%       sim.MPA.tol = 1e-4
%
%       sim.lmc.mmx_dir_path = ''
%       sim.lmc.model  = (1) input lmc.model
%                        (2) 0, if no input lmc.model
%       sim.lmc.Lcorr = inf
%       sim.lmc.Lcoup = []
%
%       sim.scalar = true;
%
%       sim.adaptive_deltaZ.model = 1;
%       sim.adaptive_deltaZ.threshold = 1e-5
%
%       sim.single_yes = true
%       sim.gpu_yes = true
%       sim.step_method = 'RK4IP'
%       sim.Raman_model = 1;
%       sim.gain_model = 0
%       sim.lmc.model = 0
%
%       sim.progress_bar = true
%       sim.progress_bar_name = ''
%       sim.compile_cuda_files = false
%       sim.check_nan = true
%       sim.verbose = false
%       sim.cuda_dir_path = [folder_of_this_function 'GMMNLSE/cuda']
%
%   <-- Correlated parameters are loaded directly -->
%
%   single-mode -- >
%
%       sim.midx = [] (not used)
%
%       Assume 1030nm for positive dispersion,
%           fiber.betas = [8.867e6; 4.903e3; 0.0208; 33.3e-6; -27.7e-9];
%           fiber.MFD = 6.2; % um; 1030nm from Thorlabs 1060XP
%       Assume 1550nm for negative dispersion,
%           fiber.betas = [5.87e6; 4.91e3; -0.0167; 1.04e-6; -324e-9];
%           fiber.MFD = 9.5; % um; 1030nm from Thorlabs 1060XP
%
%       fiber.SR = (1) 1/Aeff, if (a) there's input MFD
%                                 (b) MFD is taken from the default one and there's no input MFD
%                  (2) input SR, if there's input SR (input SR precedes over input MFD)
%       *fiber.gain_Aeff = 1/fiber.SR (taken from above)
%
%	multimode -->
%
%       fiber.MFD = [] (not used)
%
%       sim.midx = (1) []
%                  (2) input midx
%
%       fiber.betas - loaded from betars_filename in fiber.MM_folder (loaded modes are based on the above midx)
%       fiber.SR - loaded from S_tensors_filename in fiber.MM_folder (loaded modes are based on the above midx)
%       *fiber.gain_Aeff = 1/fiber.SR(1,1,1,1) (taken from above)
%
%   fiber.L0 = (1) input L0
%              (2) 2 (m)
%
%   fiber.db_gain = (1) input gain under dB
%                   (2) 30 (dB/m)
%   fiber.gain_coeff = fiber.db_gain*log(10)/(10*fiber.L0); % m^-1, converted from db
%   fiber.gain_fwhm = 40e-9; % m
%
%   *fiber.gain_tau = (1) input gain_tau
%                    (2) 840e-6; % s; 840us is the lifetime of Yb ions
%   *fiber.t_rep = (1) input t_rep
%                 (2) 1/15e6; % s; assume 15MHz repetition rate
%   *fiber.gain_cross_section = (1)input gain_cross_section
%                              (2) 6.43e-25 + 4.53e-26; % m^2; the total cross section of Yb ions at 1030nm
%
%   fiber.saturation_intensity - calculated w.r.t gain_tau, t_rep, and gain_cross_section above
%   fiber.saturation_energy - calculated w.r.t saturation_intensity and gain_Aeff above
%
%
%% ========================================================================
%   Details of each parameter
% -------------------------------------------------------------------------
%
% type_of_mode: "single_mode" or "multimode"
%
%       single_mode: read the default single-mode betas and SR tensors
%       multimode: read the MM-fiber betas and SR tensors in the "GMMNLSE/Fibers/" folder.
%       
%       [variable left empty]: default to "single_mode", like the example below 
%
%
% Example Use:
%
%    % User-defined parameters
%    fiber.betas = [0 0 0.02 0];
%    fiber.L0 = 3;
%    
%    % Incorporate default settings
%    [fiber,sim] = load_default_GMMNLSE_propagate(fiber,[]); % single_mode
%
%    % If there are "sim" settings
%    sim.fr = 0.18;
%    [fiber,sim] =  load_default_GMMNLSE_propagate(fiber,sim); % single_mode
%
%    % Use only user-defined "sim", not "fiber"
%    [fiber,sim] = load_default_GMMNLSE_propagate([],sim); % single_mode
%
%    % For multimode, add the string 'multimode' as the last argument.
%    [fiber,sim] = load_default_GMMNLSE_propagate(fiber,sim,'multimode');
%
% -------------------------------------------------------------------------
%
%	Additional parameters:
%
%       input_sim.lambda0 - central wavelength, in m
%       input_fiber.MFD - mode field diameter for calculating Aeff for SR in single mode; only used in single mode, in um
%       input_sim.midx - an array of the mode index of modes in calculations
%
%       -------------------------------------------------------------------
%       --     Explanation of "midx":
%       --         If I want only mode 2 and 4 in simulations, they should be set as
%       -- 
%       --             sim.midx = [2 4];
%       -- 
%       --         Then it's read as
%       -- 
%       --             betas = betas_from_mat_file(:,midx);
%       --             SR = SR_from_mat_file(midx,midx,midx,midx);
%       -- 
%       --         So be careful that both "betas.mat" and "S_tensor_?modes.mat" should have the same modes inside.
%       -------------------------------------------------------------------
%
%       -- input_fiber.MM_folder - the folder where betas and SRSK mat files are stored; only used in multimode
%       -- input_fiber.betas_filename - the file name of betas parameter
%       -- input_fiber.S_tensors_filename - the file name of SR tensor.
%
% -------------------------------------------------------------------------
%
%   If both "lambda0" and "f0" are set by users, the final value will depend on "f0".
%
% -------------------------------------------------------------------------
%
%   "fiber" is a structure with the fields:
%
%       Basic properties -->
%
%           betas - a (?,nm) matrix; "nm" = num_spatial_modes if under scalar fields;
%                                    otherwise, "nm" can be both num_spatial_modes or 2*num_spatial_modes depending on whether there's birefringence.
%                   betas(i, :) = (i-1)th order dispersion coefficient for each mode, in ps^n/m
%
%           attenuation - the loss of the fiber, in dB/m (default to 0)
%           n2 - the nonlinear coefficient (default to 2.3e-20 if not set)
%
%           SR - SR tensor, in m^-2
%           L0 - length of fiber, in m
%
%       Gain properties (for gain model 1,2,3; for rate-eqn gain model, see "gain_info.m") -->
%
%           db_gain - the small-signal gain amplification of the pulse energy in dB;
%                     This is used to calculate the gain_coeff  (default to 30)
%           gain_coeff - small signal gain coefficient in m^-1, defined by "g" in A(z)=exp(gz/2)A(0)
%           gain_fwhm - FWHM of the gain spectrum, in m
%
%           saturation ==>
%                   The following three parameters are used to calculate the saturation intensity,
%                   or you can set the saturation intensity directly.
%
%                   gain_tau - the lifetime of the upper energy level
%                   t_rep - the repetition rate of the pulse
%                   gain_cross_section - the absorption+emission cross sections
%
%               saturation_intensity - for Taylor or new gain model, the scale intensity in J/m^2
%                                      This is defined by h*f/(sigma*tau), where f is the center frequency,
%                                                                                sigma is the sum of the emission and absorption cross sections,
%                                                                                tau is the lifetime of the higher energy level for population inversion,
%                   OR
%               saturation_energy - for SM gain mode, the scale energy in nJ
%                                   This is defined by "saturation_intensity*Aeff"
%
% -------------------------------------------------------------------------
%
%   "initial_condition" is a structure with the fields:
%
%       dt - time step
%       fields - initial field, in W^1/2, (N-by-num_modes).
%                If the size is (N-by-num_modes-by-S), then it will take the last S.
%
%                num_modes = num_spatial_modes if "sim.scalar = true"
%                num_modes = num_spatial_modes*2 (spatial modes + polarization modes-x,y) otherwise
%
% -------------------------------------------------------------------------
%
%   "sim" is a structure with the fields:
%
%       Basic settings -->
%
%           betas - the betas for the slowly varying approximation and the moving frame, 
%                   that is to say, fiber.betas([1 2],:) = fiber.betas([1 2],:) - sim.betas;
%                   (2,1) column vector;
%                   if not set, no "sim.betas", the simulation will be run relative to the first mode
%           f0 - center frequency, in THz
%           sw - 1 includes self-steepening, 0 does not
%           deltaZ - small step size, in m
%                    This may need to be 1-50 um to account for intermodal beating,
%                    even if the nonlinear length is much larger than 50 um.
%           save_period - spatial period between saves, in m
%                         0 = only save input and output (save_period = fiber.L0)
%
%       MPA -->
%
%           MPA.M - parallel extent for MPA;
%                   1 is no parallelization,
%                   5-20 is recommended; there are strongly diminishing returns after 5-10.
%           MPA.n_tot_max - maximum number of iterations for MPA
%                           This doesn't really matter because if the step size is too large, the algorithm will diverge after a few iterations.
%           MPA.n_tot_min - minimum number of iterations for MPA
%           MPA.tol - tolerance for convergence for MPA
%                     Value of the average NRMSE between consecutive itertaions in MPA at which the step is considered converged.
%
%       Linear mode coupling -->
%
%           To run with linear mode coupling,
%           "mmx" is required for CPU computation; otherwise, GPU is required.
%
%           lmc.mmx_dir_path - path to the "mmx" directory;
%                              This is used with linear mode coupling;
%                              "mmx" is a MATLAB function for matrix operations page by page, like "pagefun" for GPU,
%                              so it's much faster than using for-loop.
%                              Highly recommend the user to install this plugin for CPU computing.
%                              If it's an empty string, it's not set.
%
%           lmc.model - 0 = no linear mode coupling
%                       1 = unitary-matrix-R model
%                       2 = iQA-addition model
%                       3 = Manakov eqaution
%
%           Manakov equation ==>
%
%               When lmc.model = 3
%               Use "mode_groups" below.
%                   If this term is empty or doesn't exist in "sim", it's assumed that they're all in the same mode group.
%
%           lmc.Lcorr - any number; the correlation length of random mode coupling
%                       inf = use the same random matrix along the fiber
%
%           lmc.Lcoup - (lmc.model: 2) a matrix of the size (num_mode_groups,num_mode_groups);
%                       the coupling length of linear mode coupling between or inside mode groups, in m
%                       "inf" for its elements = no random mode coupling for that linear inter-or-intra-mode-group coupling
%
%           ---------------------------------------------------------------
%
%           lmc.mode_groups - an array; each number representing the number of modes in that mode groups.
%                             The summation of this array should be equal to the total number of modes.
%
%                                 E.g. [2 4] means there are 6 modes in total and 2 mode groups
%
%                             For lmc_model 1,2, this determines the block-diagonal structure of the random matrices.
%
%                             If not specified (no field "mode_group" in "sim" or sim.mode_groups=[];), 
%                                 R model:   the structure of R is determined by the difference in group velocities.
%                                 iQA model: Q is a full random matrix
%
%       Polarization included -->
%
%           scalar - 0(false) = consider polarization mode coupling
%                    1(true) = don't consider polarization mode coupling
%
%           *If under scalar field, the input field takes only the scalar fields, e.g., [mode1, mode2, mode3......].
%           *Otherwise, the input field of each polarization needs to be specified in the order of [mode1_+ mode1_- mode2_+ mode2_-......], where (+,-) can be (x,y) or any orthogonal modes.
%           *SRSK is always loaded in a dimension of num_spatial_modes^4. It's automatically calculated to its polarized version in the code.
%
%           ellipticity - the ellipticity of the polarization modes; Please refer to "Nonlinear Fiber Optics, eq (6.1.18) Agrawal" for the equations.
%                         0: linear polarization   -> (+,-)=(x,y)
%                         1: circular polarization -> (+,-)=(right,left)
%
%       Adaptive method -->
%
%           adaptive_deltaZ.model - 0 = don't use adaptive step method,
%                                   1 = use adaptive step based on RK4IP
%           adaptive_deltaZ.threshold - a scalar;
%                                       the accuracy used to determined whether to increase or decrease the step size.
%
%           Currently, adaptive method supports only RK4IP.
%
%       Algorithms to use -->
%
%           singe_yes - 1(true) = single
%                       0(false) = double
%
%                       Many GPUs are optimized for single precision, to the point where one can get a 10-30x speedup just by switching to single precision.
%
%           gpu_yes - 1(true) = GPU
%                     0(false) = CPU
%
%                     Whether or not to use the GPU. Using the GPU is HIGHLY recommended, as a speedup of 50-100x should be possible.
%
%           Raman_model - 0 = ignore Raman effect
%                         1 = Raman model approximated analytically by a single vibrational frequency of silica molecules
%                                 (Ch. 2.3, p.42, Nonlinear Fiber Optics (5th), Agrawal)
%                         2 = Raman model including the anisotropic contribution
%                                 ("Ch. 2.3, p.43" and "Ch. 8.5, p.340", Nonlinear Fiber Optics (5th), Agrawal)
%                                 For more details, please read "Raman response function for silica fibers", by Q. Lin and Govind P. Agrawal (2006)
%
%           step_method - a string that specifies which pulse-propagation algorithm to use;
%                         'split-step' - split-step algorithm
%                         'RK4IP'      - Runge-kutta under the interaction picture
%                         'MPA'        - massively parallel algorithm
%
%           gain_model - 0 = no gain
%                        1 = SM gain (total energy saturation)
%                        2 = new gain (spatial saturation)
%                        3 = Taylor expansion gain (approximate spatial saturation)
%                        4 = Gain-rate-equation model: see "gain_info.m" for details
%
%       Others -->
%
%           gpuDevice.Index - a scalar; the GPU to use
%           gpuDevice.Device - the output of MATLAB "gpuDevice(gpu_index)"
%           check_nan  - 1(true) = Check if the field has NaN components each roundtrip, 0(false) = do not
%           cuda_dir_path - path to the cuda directory into which ptx files will be compiled and stored
%           progress_bar - 1(true) = show progress bar, 0(false) = do not
%                          It'll slow down the code slightly. Turn it off for performance.
%           progress_bar_name - the name of the GMMNLSE propagation shown on the progress bar.
%                               If not set (no "sim.progress_bar_name"), it uses a default empty string, ''.
%
% =========================================================================

%% Current path (or the folder where this "load_default_GMMNLSE_propagate.m" is)
if ispc
    sep = '\';
else % unix
    sep = '/';
end
current_path = mfilename('fullpath');
sep_pos = strfind(current_path,sep);
upper_folder = current_path(1:sep_pos(end-1));

%% Default settings below:

% Supperss warnings generated by the function, "catstruct", due to there
% are definitely duplicate elements between default and input.
warning('off','catstruct:DuplicatesFound');

if ~exist('input_fiber','var')
    input_fiber = [];
end
if ~exist('input_sim','var')
    input_sim = [];
end

if exist('type_of_mode','var')
    if ~any(strcmp(type_of_mode,{'single_mode','multimode'}))
        error('LoadDefaultGMMNLSEPropagate:TypeOfModeError',...
            'It should be either "single_mode" or "multimode".');
    end
else
    % Set default to "single mode"
    type_of_mode = 'single_mode';
end

% -------------------------------------------------------------------------
% Set some default parameters early here because these parameters will be
% used for loading files if multimode.
% If single-mode, only "fiber.lambda0" or "fiber.f0" is important.
% -------------------------------------------------------------------------
c = 2.99792458e-4; % speed of ligth, m/ps

% Get lambda0 from input f0 or lambda0
if isfield(input_sim,'f0')
    default_sim.lambda0 = c/input_sim.f0;
else
    if isfield(input_sim,'lambda0')
        default_sim.lambda0 = input_sim.lambda0;
    else
        default_sim.lambda0 = 1030e-9;
    end
end

% -------------------------------------------------------------------------
% fiber
% -------------------------------------------------------------------------
% Basic properties
% (Load betas, SR tensors, and Aeff below)
switch type_of_mode
    case 'single_mode'
        default_sim.midx = [];
        
        if default_sim.lambda0 < 1.3e-6 % lambda(zero dispersion)=1.3um; assume 1030nm for positive dispersion
            default_fiber.betas = [8.867e6; 4.903e3; 0.0208; 33.3e-6; -27.7e-9];
            default_fiber.MFD = 6.2; % um; 1030nm from Thorlabs 1060XP
        else % assume 1550nm for negative dispersion
            default_fiber.betas = [5.87e6; 4.91e3; -0.0167; 1.04e-6; -324e-9];
            default_fiber.MFD = 9.5; % um; 1030nm from Thorlabs 1060XP
        end
        % Load "input_fiber" into "default_fiber" first to calculate SR.
        if isfield(input_fiber,'MFD')
            default_fiber.MFD = input_fiber.MFD;
        end
        
        Aeff = pi*(default_fiber.MFD/2)^2*1e-12; % effective area of the SMF, [m^2]
        default_fiber.SR = 1/Aeff;
        
        % Because MFD is used to calculate SR for single-mode case,
        % clear it if input_fiber has SR already.
        if isfield(input_fiber,'SR')
            if isfield(input_sim,'gain_model') && ~ismember(input_sim.gain_model,[2 3]) % Aeff is necessary for calculating gain for gain_model 2 and 3, but these two models are typically used only for multimode. Check it here just to be safe.
                default_fiber.MFD = [];
            else
                Aeff = 1./input_fiber.SR;
            end
        end
        default_fiber.gain_Aeff = Aeff;
    case 'multimode'
        just_load_files = ~isfield(input_sim,'midx') || isempty(input_sim.midx);
        
        if just_load_files
            default_sim.midx = [];
        else
            default_sim.midx = input_sim.midx;
        end
        
        % betas
        if ~isfield(input_fiber,'betas') || isempty(input_fiber.betas)
            try
                load([input_fiber.MM_folder input_fiber.betas_filename],'betas'); % in fs^n/mm
            catch
                error('load_default_GMMNLSE_propagate:LoadError',...
                      'Please check if the folder name is correct.\nUse "fiber.MM_folder, fiber.betas_filename, or fiber.S_tensors_filename" to specify the target folder or filename.');
            end
            unit_conversion = 0.001.^(-1:size(betas, 1)-2)'; % The imported values are in fs^n/mm, but the simulation uses ps^n/m
            if just_load_files
                default_fiber.betas = betas.*unit_conversion;
            else
                default_fiber.betas = betas(:,default_sim.midx).*unit_conversion;
            end
        end
        
        % SR
        % It needs to be taken from input_fiber first because Aeff is used in calculation of saturation_intensity for "new gain model" later.
        if ~isfield(input_fiber,'SR') || isempty(input_fiber.SR)
            try
                load([input_fiber.MM_folder input_fiber.S_tensors_filename],'SR'); % in m^-2
            catch
                error('load_default_GMMNLSE_propagate:LoadError',...
                'Please check if the folder name is correct.\nUse "fiber.MM_folder, fiber.betas_filename, or fiber.S_tensors_filename" to specify the target folder or filename.');
            end
        
            if just_load_files
                default_fiber.SR = SR;
            else
                midx = default_sim.midx;
                default_fiber.SR = SR(midx,midx,midx,midx,:);
            end
        else
            default_fiber.SR = input_fiber.SR;
        end
        Aeff = 1./default_fiber.SR(1,1,1,1,:); % get Aeff from the fundamental mode first (important: SR needs to start with mode 1)
        
        default_fiber.MFD = []; % not used, for the consistency of the output between different inputs
        default_fiber.gain_Aeff = Aeff;
end

% L0 is necessary to be put into "default_fiber" first for "gain_coeff" calculation.
if isfield(input_fiber,'L0')
    default_fiber.L0 = input_fiber.L0;
else
    default_fiber.L0 = 2; % m
end

% Gain properties
if isfield(input_sim,'gain_model') && ismember(input_sim.gain_model,[1,2,3])
    if isfield(input_fiber,'db_gain') % total small signal gain in the gain fiber
        default_fiber.db_gain = input_fiber.db_gain; % Because we need db_gain to calculate gain_coeff, we need to load the user's one first if it exists
    else
        default_fiber.db_gain = 30;
    end
    default_fiber.gain_coeff = default_fiber.db_gain*log(10)/(10*default_fiber.L0); % m^-1, converted from db
    default_fiber.gain_fwhm = 40e-9; % m
    
    % saturation energy/intensity
    if isfield(input_fiber,'gain_tau') && ~isempty(input_fiber.gain_tau)
        default_fiber.gain_tau = input_fiber.gain_tau;
    else
        default_fiber.gain_tau = 840e-6; % s; 840us is the lifetime of Yb ions
    end
    if isfield(input_fiber,'t_rep') && ~isempty(input_fiber.t_rep)
        default_fiber.t_rep = input_fiber.t_rep;
    else
        default_fiber.t_rep = 1/15e6; % s; assume 15MHz repetition rate
    end
    if isfield(input_fiber,'gain_cross_section') && ~isempty(input_fiber.gain_cross_section)
        default_fiber.gain_cross_section = input_fiber.gain_cross_section;
    else
        default_fiber.gain_cross_section = 6.43e-25 + 4.53e-26; % m^2; the total cross section of Yb ions at 1030nm
    end
    h = 6.626e-34; % Planck constant
    default_fiber.saturation_intensity = h*(c*1e12)/default_sim.lambda0/default_fiber.gain_cross_section/default_fiber.gain_tau*default_fiber.t_rep; % J/m^2
    default_fiber.saturation_energy = default_fiber.saturation_intensity*default_fiber.gain_Aeff*1e9; % nJ
else % no gain or rate-equation gain model;
     % not used, for the consistency of the output between different inputs
    default_fiber.gain_coeff = [];
    default_fiber.gain_fwhm = [];
    default_fiber.saturation_energy = [];
    default_fiber.saturation_intensity = [];
end

% -------------------------------------------------------------------------
% sim
% -------------------------------------------------------------------------
% Basic settings
default_sim.f0 = c/default_sim.lambda0; % THz
default_sim.sw = true; % shock term
default_sim.deltaZ = 250e-6; % m
default_sim.save_period = 0; % m
default_sim.ellipticity = 0; % linear polarization

% MPA
default_sim.MPA.M = 10;
default_sim.MPA.n_tot_max = 20;
default_sim.MPA.n_tot_min = 2;
default_sim.MPA.tol = 1e-4;

% Linear mode coupling
default_sim.lmc.mmx_dir_path = '';
default_sim.lmc.Lcorr = inf;
default_sim.lmc.Lcoup = [];

% Polarization modes
default_sim.scalar = true;

% Adaptive method
default_sim.adaptive_deltaZ.threshold = 1e-5; % the threshold of the adaptive method
                                              % Recommended value is 1e-5. Values larger than 1e-3 are too large.

% Algorithms to use
default_sim.adaptive_deltaZ.model = 1; % Use "adaptive step size" method
default_sim.single_yes = true;
default_sim.gpu_yes = true;
default_sim.step_method = 'RK4IP';
if isfield(input_sim,'Raman_model')
   default_sim.Raman_model = input_sim.Raman_model;
else
    default_sim.Raman_model = 1;
end
% default_sim.Raman_model = 1; % isotropic Raman model
default_sim.gain_model = 0;
default_sim.lmc.model = 0;

% Others
default_sim.gpuDevice.Index = 1; % the gpuDevice to use
default_sim.progress_bar = true;
default_sim.progress_bar_name = '';
default_sim.compile_cuda_files = false;
default_sim.check_nan = true;
default_sim.verbose = false;
default_sim.cuda_dir_path = fullfile(upper_folder,'GMMNLSE','cuda');

%%
% =========================================================================
% Merge settings with the input, which have higher priorities than the
% default ones.
% =========================================================================
if isempty(input_fiber)
    fiber = default_fiber;
elseif isstruct(input_fiber)
    fiber = catstruct(default_fiber, input_fiber);
else
    error('LoadDefaultGMMNLSEPropagate:InputFiberError',...
            '"input_fiber" should be a "structure".');
end
if isempty(input_sim)
    sim = default_sim;
elseif isstruct(input_sim)
    sim = catstruct(default_sim, input_sim);
else
    error('LoadDefaultGMMNLSEPropagate:InputSimError',...
            '"input_sim" should be a "structure".');
end

end