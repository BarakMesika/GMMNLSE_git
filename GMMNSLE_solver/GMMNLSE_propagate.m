function foutput = GMMNLSE_propagate(fiber, initial_condition, sim, rate_gain_parameters,rate_gain_saved_data)
%GMMNLSE_PROPAGATE Propagate an initial multimode pulse through an arbitrary distance of an optical fiber
%   This is a caller function, calling
%   GMMNLSE_propagate_with_adaptive_deltaZ or
%   GMMNLSE_propagate_no_adaptive_deltaZ 
%   based on whether to use adaptive step size method or not.
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
%           gain_coeff - small signal gain coefficient in m^-1, defined by A(z)=exp(gz/2)A(0)
%           gain_fwhm - FWHM of the gain spectrum, in m
%
%           saturation ==>
%               saturation_intensity - for Taylor or new gain model, the scale intensity in J/m^2
%                   OR
%               saturation_energy - for SM gain mode, the scale energy in nJ
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
%                             For lmc.model 1,2, this determines the block-diagonal structure of the random matrices.
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
%                                   1 = use adaptive step based on RK4IP,
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
%           gpuDevice - the output of MATLAB "gpuDevice()"
%           check_nan  - 1(true) = Check if the field has NaN components each roundtrip, 0(false) = do not
%           cuda_dir_path - path to the cuda directory into which ptx files will be compiled and stored
%           progress_bar - 1(true) = show progress bar, 0(false) = do not
%                          It'll slow down the code slightly. Turn it off for performance.
%           progress_bar_name - the name of the GMMNLSE propagation shown on the progress bar.
%                               If not set (no "sim.progress_bar_name"), it uses a default empty string, ''.
%
%
% -------------------------------------------------------------------------
%
%   gain_parameters - Only used for gain_model=4, "gain rate-equation model".
%                     Please refer to "gain_info.m" for details.
%
% =========================================================================
%
% All the cases of num_modes for input fields and betas:
%
%                 | field   betas   SRSK
%   --------------------------------------------
%   (s,nl)        |   m       m      m
%   (s,M)         |   m       m      m
%   (s,r)         |   m       m      m
%   (p,nl)        |   2m     m,2m    m
%   (p,M)         |   2m     m,2m    m
%   (p,r)         |   2m     m,2m    m
%
%   m: num_spatial_modes
%   s: scalar fields, p: polarized fields
%   nl: no linear mode coupling, M: Manakov equation, r: random linear coupling
%
%   If num_modes(betas) = m, it's assumed that polarization modes are
%   degenerate in betas and is expanded into 2m modes automatically in 
%   the code, that is, (m,2m)->2m.
%
%   SRSK is always in a dimension of num_spatial_modes^4.
%
% =========================================================================
% Output:
% foutput.fields - (N, num_modes, num_save_points) matrix with the multimode field at each save point
% foutput.dt - time grid point spacing, to fully identify the field
% foutput.z - the propagation length of the saved points
% foutput.deltaZ - the (small) step size for each saved points
% foutput.betas - the [betas0,betas1] used for the moving frame
% foutput.t_delay - the time delay of each pulse which is centered in the time window during propagation
% foutput.seconds - time spent in the main loop
%
% For MPA:
%   foutput.full_iterations_hist - histogram of the number of iterations, accumulated and saved between each save point
%
% For gain-rate-equation model:
%   foutput.Power.pump_forward - (1,1,num_save_points); the forward pump power along the fiber
%   foutput.Power.pump_backward - (1,1,num_save_points); the backward pump power along the fiber
%   *If ASE is considered,
%       foutput.Power.ASE_forward - (1,1,num_save_points); the forward ASE power along the fiber
%       foutput.Power.ASE_backward - (1,1,num_save_points); the backward ASE power along the fiber
%   *If N2 is exported,
%       foutput.N2 - (Nx,Nx,num_save_points); the doped ion density of the upper state
%   *If reuse_data,
%       If save_all_in_RAM,
%           foutput.saved_data - saved_data_info used for an oscillator to converge faster
%       else
%           foutput.clean_rate_gain_mat - the function handle to clear the saved mat files used for an oscillator to converge faster

%%
if ispc
    sep_char = '\';
else % unix
    sep_char = '/';
end

%% If using CPU, use "mmx" by Yuval on MATLAB File Exchange
% "mmx" function runs several matrix operations page by page which is the
% same as "pagefun" for GPU, so it's much faster than using for-loop.
% Highly recommend users to install this plugin for CPU computing.
sim.mmx_yes = false;
if ~sim.gpu_yes % CPU
    if isfield(sim,'mmx_dir_path') && ~isempty(sim.mmx_dir_path)
        addpath(sim.mmx_dir_path);
        if exist('mmx','file') && exist('mmx_mult','file') && exist('multbslash','file') && exist('multslash','file') && exist('multinv','file') && exist('mpower2','file')
            sim.mmx_yes = true;
        end
    end
end

% Load the folder
current_path = mfilename('fullpath');
sep_pos = strfind(current_path,sep_char);
current_folder = current_path(1:sep_pos(end));
addpath([current_folder 'GMMNLSE algorithm/']);

%%
switch sim.adaptive_deltaZ.model
    case 0
        if sim.gain_model == 4
            if ~exist('rate_gain_saved_data','var')
                rate_gain_saved_data = []; % create a dummy variable to run the function
            end

            foutput = GMMNLSE_propagate_no_adaptive_deltaZ(fiber, initial_condition, sim, rate_gain_parameters, rate_gain_saved_data);
        else
            foutput = GMMNLSE_propagate_no_adaptive_deltaZ(fiber, initial_condition, sim);
        end
    case 1
        if sim.lmc.model ~= 0
            error('GMMNLSE_propagate:modelError',...
                  'Adaptive step algorithm doesn''t work for linear mode coupling.');
        end
        
        if sim.gain_model == 4
            foutput = GMMNLSE_propagate_with_adaptive_deltaZ_RK4IP(fiber, initial_condition, sim, rate_gain_parameters);
        else
            foutput = GMMNLSE_propagate_with_adaptive_deltaZ_RK4IP(fiber, initial_condition, sim);
        end
end