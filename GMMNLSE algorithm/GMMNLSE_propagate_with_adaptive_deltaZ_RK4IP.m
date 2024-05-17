function foutput = GMMNLSE_propagate_with_adaptive_deltaZ_RK4IP(fiber, initial_condition, sim, rate_gain_parameters)
%GMMNLSE_PROPAGATE_WITH_ADAPTIVE_DELTAZ_RK4IP Propagate an initial multimode
%pulse through an arbitrary distance of an optical fiber with the use of
%adaptive step method.
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
%    s            |   m       m      m
%    p            |   2m     m,2m    m
%
%   m: num_spatial_modes
%   s: scalar fields, p: polarized fields
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
% For gain-rate-equation model:
%   foutput.Power.pump_forward - (1,1,num_save_points); the forward pump power along the fiber
%   foutput.Power.pump_backward - (1,1,num_save_points); the backward pump power along the fiber
%   *If N2 is exported,
%       foutput.N2 - (Nx,Nx,num_save_points); the doped ion density of the upper state

%% Check the validity of input parameters
if sim.save_period == 0
    sim.save_period = fiber.L0;
end

num_saves_total = fiber.L0/sim.save_period;
if rem(num_saves_total,1) && rem(num_saves_total+eps(num_saves_total),1) && rem(num_saves_total-eps(num_saves_total),1)
    error('GMMNLSE_propagate:SizeIncommemsurateError',...
          'The save period is %f m and the fiber length is %f m, which are not commensurate', sim.save_period, fiber.L0)
else
    num_saves_total = round(num_saves_total);
end

% Error check on the dimensions (num_modes) of matrices
check_nummodes(sim, fiber, initial_condition.fields);

%% Check the precision of the input fields and match it with "sim.single_yes"
if isequal(class(initial_condition.fields),'gpuArray')
    input_field_precision = classUnderlying(initial_condition.fields);
else
    input_field_precision = class(initial_condition.fields);
end
if sim.single_yes
    running_precision = 'single';
else
    running_precision = 'double';
end
if ~isequal(input_field_precision,running_precision)
    initial_condition.fields = feval(str2func(['@' running_precision]), initial_condition.fields);
end

%% Some Parameters

% Get the numerical parameters from the initial condition.
[Nt, num_modes,~] = size(initial_condition.fields);
num_spatial_modes = size(fiber.SR,1);

% For polarized fields, the dimension of the input betas is allowed to be
% "num_spatial_modes", so it needs to be expanded into
% "2*num_spatial_modes" in the rest of the computation.
fiber = betas_expansion_including_polarization_modes(sim,fiber,num_modes);

%% Pulse centering based on the moment of its intensity
% Center the pulse
tCenter = floor(sum(sum((-floor(Nt/2):floor((Nt-1)/2))'.*abs(initial_condition.fields).^2),2)/sum(sum(abs(initial_condition.fields).^2),2));
% Because circshift is slow on GPU, I discard it.
%last_result = ifft(circshift(initial_condition.fields,-tCenter));
if tCenter ~= 0
    if tCenter > 0
        initial_condition.fields = [initial_condition.fields(1+tCenter:end,:);initial_condition.fields(1:tCenter,:)];
    elseif tCenter < 0
        initial_condition.fields = [initial_condition.fields(end+1+tCenter:end,:);initial_condition.fields(1:end+tCenter,:)];
    end

    if sim.gpu_yes
        tCenter = gather(tCenter);
    end
    t_delay = tCenter*initial_condition.dt;
else
    t_delay = 0;
end

%% Set up the GPU details

    % ---------------------------------------------------------------------
    function recompile_ptx(cudaFilename,ptxFilename)
        if ispc
            system(['nvcc -ptx "', fullfile(sim.cuda_dir_path,cudaFilename), '" --output-file "', fullfile(sim.cuda_dir_path,ptxFilename) '"']);
        else % unix
            % tested: Debian 9 (Stretch)
            % Cuda 8 doesn't support gcc6, beware to use gcc5 or clang-3.8.
            system(['nvcc -ccbin clang-3.8 -ptx "', fullfile(sim.cuda_dir_path,cudaFilename), '" --output-file "', fullfile(sim.cuda_dir_path,ptxFilename) '"']);
        end
    end
    % ---------------------------------------------------------------------

% Use the specified GPU
% This needs to run at the beginning; otherwise, the already-stored values
% in GPU will be unavailable in a new GPU if the GPU device is switched.
if sim.gpu_yes
    try
        sim.gpuDevice.Device = gpuDevice(sim.gpuDevice.Index); % use the specified GPU device
    catch
        error('Please set the GPU you''re going to use by setting "sim.gpuDevice.Index".');
    end
end

if sim.gpu_yes
    if sim.single_yes
        num_size = 8; % 4 bytes * 2 for complex
        single_str = 'single';
    else
        num_size = 16; % 8 bytes * 2 for complex
        single_str = 'double';
    end
    
    % For no gain, SM gain, or rate-eqn gain use the normal cuda file
    % For the new gain model, use the slightly modified cuda file
    if ismember(sim.gain_model, [0 1 4])
        fname_part = 'sumterm';
    else % sim.gain_model == [2 3]; new_gain/taylor_gain models
        fname_part = 'sumterms';
    end
    
    % Polarization modes
    if sim.scalar
        polar_str = '';
    else % polarized fields or scalar Manakov
        polar_str = 'polarization_';
    end
    
    % Whether to include "anisotropic Raman term" or not
    if sim.Raman_model==0
        Raman_str = 'noRaman_';
    elseif ~sim.scalar && sim.Raman_model==2
        Raman_str = 'anisoRaman_';
    else
        Raman_str = '';
    end
    
    % Nonlinear term
    specific_filename = ['calculate_' fname_part '_part_' polar_str Raman_str single_str];
    cudaFilename = [specific_filename, '.cu'];
    ptxFilename = [specific_filename, '.ptx'];
    
    if ~exist(fullfile(sim.cuda_dir_path,ptxFilename), 'file')
        recompile_ptx(cudaFilename,ptxFilename);
    end
    
    % Setup the kernel from the cu and ptx files
    try
        kernel = parallel.gpu.CUDAKernel(fullfile(sim.cuda_dir_path,ptxFilename), fullfile(sim.cuda_dir_path,cudaFilename));
    catch
        % Compile the CUDA code again.
        % Currently found error:
        %    version mismatch due to different versions of cuda I use in Windows and Debian.
        recompile_ptx(cudaFilename,ptxFilename);
        kernel = parallel.gpu.CUDAKernel(fullfile(sim.cuda_dir_path,ptxFilename), fullfile(sim.cuda_dir_path,cudaFilename));
    end
    
    % Break up the computation into threads, and group them into blocks
    num_threads_per_block = floor(sim.gpuDevice.Device.MaxShmemPerBlock/(num_modes*(2+num_modes)*num_size));
    if num_threads_per_block < 32
        num_threads_per_block = 32;
    end
    if num_threads_per_block > sim.gpuDevice.Device.MaxThreadBlockSize(1)
        num_threads_per_block = sim.gpuDevice.Device.MaxThreadBlockSize(1);
    end
    
    num_blocks = ceil(Nt/num_threads_per_block);
    
    kernel.ThreadBlockSize = [num_threads_per_block,1,1];
    kernel.GridSize = [num_blocks,1,1];
    
    % Finally save the kernel
    sim.kernel = kernel;
end

%% Work out the overlap tensor details
% Because of the symmetry, there are possibly zeros in SR. To improve the
% performance, store only the indices of each nonzero elements and their
% corresponding values.
nonzero_midx1234s = find(permute(fiber.SR,[4 3 2 1])); % The indices of nonzero elements in SR will be stored in nonzero_midx1234s.
nonzero_midx1234s = reverse_linear_indexing(nonzero_midx1234s,num_spatial_modes);
[midx1,midx2,midx3,midx4] = ind2sub(num_spatial_modes*ones(1,4),nonzero_midx1234s'); % restore linear indexing back to subscripts
SRa_info.nonzero_midx1234s = uint8([midx1;midx2;midx3;midx4]);
SRa_info.SRa = fiber.SR(nonzero_midx1234s); % the corresponding values of each nonzero elements

if sim.single_yes
   SRa_info.SRa = single(SRa_info.SRa);
end
if sim.gpu_yes
   SRa_info.SRa = gpuArray(SRa_info.SRa);
else % If not using the GPU, we also need to calculate the indices that don't have all zero coefficients for any given last two indices
    tmp = cellfun(@(SR12) any(SR12(:)), mat2cell(fiber.SR,num_spatial_modes,num_spatial_modes,ones(num_spatial_modes,1),ones(num_spatial_modes,1)));
    nonzero_midx34s = find(squeeze(tmp));
    [midx3,midx4] = ind2sub(num_spatial_modes*ones(1,2),nonzero_midx34s');
    SRa_info.nonzero_midx34s = uint8([midx3;midx4]);
end

% Update SRa, SRb, and SK regarding polarization modes.
if ~sim.scalar
    if sim.Raman_model ~= 2
        [SRa_info, SK_info] = calc_polarized_SRSK(SRa_info,sim.ellipticity);
        SRb_info = [];
    else
        [SRa_info, SRb_info, SK_info] = calc_polarized_SRSK(SRa_info,sim.ellipticity,true);
    end
else % scalar fields: SK is proportional to SR by a factor, sim.SK_factor
    switch sim.ellipticity
        case 0 % linear polarization
            sim.SK_factor = 1;
        case 1 % circular polarization
            sim.SK_factor = 2/3;
        otherwise
            error('GMMNLSE_propagate:ellipticityError',...
                'The scalar mode supports only linear and circular polarizations.');
    end
    if sim.single_yes
        sim.SK_factor = single(sim.SK_factor);
    end
    
    SK_info = [];
    SRb_info = [];
end

%% Calculate the nonlinearity constant
c = 2.99792458e-4; % speed of ligth m/ps
w0 = 2*pi*sim.f0; % angular frequency (THz)
if ~isfield(fiber,'n2') || isempty(fiber.n2)
    fiber.n2 = 2.3e-20; % m^2 W^-1
end
nonlin_const = fiber.n2*w0/c; % W^-1 m

%% Pre-calculate the dispersion term
% The "omegas" here is actually (omega - omega0), omega: true angular frequency
%                                                 omega0: central angular frequency (=2*pi*f0)
if sim.gpu_yes
    dt = gpuArray(initial_condition.dt);
else
    dt = initial_condition.dt;
end
omegas = 2*pi*ifftshift(linspace(-floor(Nt/2), floor((Nt-1)/2), Nt))'/(Nt*dt); % in 1/ps, in the order that the fft gives
if sim.single_yes
    omegas = single(omegas);
end
if sim.gpu_yes
    omegas = gpuArray(omegas);
end

% The dispersion term in the GMMNLSE, in frequency space
if any(size(fiber.betas) == Nt) % the betas is given over different frequencies,
                               % instead of the coefficients of the Taylor series expansion over the center frequency
    if size(fiber.betas,2) == Nt % betas should be a column vector
        fiber.betas = fiber.betas.';
    end
    if ~isfield(sim,'betas')
        sim.betas = zeros(2,1,'gpuArray');
        
        % Obtain the betas of the input pulse
        fftshift_omegas = fftshift(omegas);
        fit_order = 7;
        [betas_Taylor_coeff,~,mu] = polyfit(fftshift_omegas,real(fiber.betas(:,1)),fit_order);
        sim.betas = [betas_Taylor_coeff(end);betas_Taylor_coeff(end-1)];
        sim.betas = real(sim.betas);
        new_betas = real(fiber.betas(:,1))-(sim.betas(1)+sim.betas(2)*(fftshift_omegas-mu(1))/mu(2));
        sim.betas = [sim.betas(1)-sim.betas(2)*mu(1)/mu(2) + (max(new_betas) + min(new_betas))/2;...
                     sim.betas(2)/mu(2)];
    end
    
    D_op = 1i*(ifftshift(fiber.betas)-(sim.betas(1)+sim.betas(2)*omegas));
else
    % D0_op = sum( i*beta_n/n!*omega^n ,n)
    if ~isfield(sim,'betas')
        sim.betas = real(fiber.betas([1 2],1));
    end
    fiber.betas([1 2],:) = fiber.betas([1 2],:) - sim.betas; % beta0 and beta1 are set relative to sim.betas, or the fundamental mode if sim.betas doesn't exist
    % D_op = sum( 1i*beta_n/n!*omegas^n ,n,0,size(fiber.betas,1)-1 )
    taylor_n = permute(0:size(fiber.betas,1)-1,[1 3 2]); % calculation starting here is under the dimension (N,num_modes,order_betas)
    taylor_power = omegas.^taylor_n;
    D_op = sum( 1i./factorial(taylor_n).*permute(fiber.betas,[3 2 1]).*taylor_power,3); % sum(...,3) sums over different "n" adding all Taylor series expansion terms
end
if ~isfield(fiber,'attenuation')
    fiber.attenuation = 0; % in dB/m
end
D_op = D_op + log(10.^(-ifftshift(fiber.attenuation)/10));

%% Pre-calculate the factor used in GMMNLSE
prefactor = 1i*nonlin_const*(1+sim.sw*omegas/(2*pi*sim.f0));

%% Pre-calculate the gain term if necessary
if sim.gpu_yes
    switch sim.gain_model
        case 1
            fiber.saturation_energy = gpuArray(fiber.saturation_energy);
        case {2,3}
            fiber.saturation_intensity = gpuArray(fiber.saturation_intensity);
    end
end
if ismember(sim.gain_model,[1 2 3])
    % Gaussian gain
    if fiber.gain_coeff ~= 0
        w_fwhm = 2*pi*sim.f0^2/c*fiber.gain_fwhm;
        w_0 = w_fwhm/(2*sqrt(log(2))); % 2*sqrt(log(2))=1.665
        G = fiber.gain_coeff/2*exp(-omegas.^2/w_0^2);
    end
end

%% Pre-compute the Raman response in frequency space
if ~isfield(fiber,'fiber_type')
    fiber.fiber_type = 'silica';
end
[fiber,haw,hbw] = Raman_model( fiber,sim,Nt,dt);

%% Incoporate (1-fiber.fr) and fiber.fr into SRa,SRb,SK, or kappaK,kappaR1,kappaR2
if isempty(SK_info) % scalar fields
    if sim.Raman_model ~= 0
        sim.SK_factor = (1-fiber.fr)/fiber.fr*sim.SK_factor; % For scalar fields, SK is calculated from SRa.
                                                         % Because SRa will be multiplied by fiber.fr below and SK also needs to be multiplied by (1-fiber.fr), a factor of (1-fiber.fr)/fiber.fr is included here.
    end
else % polarized fields
    SK_info.SK = (1-fiber.fr)*SK_info.SK;
end

if sim.Raman_model ~= 0
    SRa_info.SRa = fiber.fr*SRa_info.SRa;
end

if ~isempty(SRb_info)
    SRb_info.SRb = fiber.fr*SRb_info.SRb;
end

%% Setup the exact save points

% We will always save the initial condition as well
save_points = int64(num_saves_total + 1);
save_z = double(0:save_points-1)'*sim.save_period;

save_deltaZ = zeros(save_points-1,1);

A_out = zeros(Nt, num_modes, save_points);
if sim.single_yes
    A_out = single(A_out);
end

% Start by saving the initial condition
A_out(:, :, 1) = initial_condition.fields(:, :, end);

t_delay_out = zeros(save_points,1);
t_delay_out(1) = t_delay;

% Also setup the last_result in the frequency domain
% This gets passed to the step function, so if using the GPU it also needs
% to live on the GPU
initial_condition.fields = ifft(initial_condition.fields(:, :, end));
if sim.gpu_yes
    initial_condition.fields = gpuArray(initial_condition.fields);
end

%% Include the shot noise
% The shot noise SNR = sqrt(N), N is the number of photons
% SNR = P_signal/P_noise = sqrt(P_signal/(2*h*f*df)), 2 stands for positive and negative frequencies
% P_noise = sqrt(2*h*f*df*P_signal)
h = 6.62607015e-34; % J*s
photon_noise = 2*h*(omegas/2/pi+sim.f0)*1e12/(Nt*dt*1e-12);
for midx = 1:size(initial_condition.fields,2) % x, y polarization
    smaller_than_photon_noise = abs(initial_condition.fields(:,midx)).^2 < photon_noise;
    initial_condition.fields( smaller_than_photon_noise,midx ) = sqrt( photon_noise(smaller_than_photon_noise) );
end
rand_real = rand(Nt,1)*2 - 1;
rand_imag = (2*randi([0,1],Nt,1)-1).*sqrt(1-rand_real.^2);
initial_condition.fields = initial_condition.fields + (rand_real+1i*rand_imag).*(photon_noise.*abs(initial_condition.fields).^2).^(1/4);

% Make the field around the edges of the time window zero to avoid aliasing
% due to the (inverse) Fourier Transform.
min_dB = -200;
DW = create_damped_window(Nt,0.2,1e-2,min_dB);
initial_condition.fields = ifft(fft(initial_condition.fields).*DW);

%% Run the step function over each step
run_start = tic;
% -------------------------------------------------------------------------
if sim.gain_model == 4 % Gain rate-equation model
    if isequal(rate_gain_parameters{1}.midx,1) % fundamental mode
        if length(rate_gain_parameters) == 7 % single-mode computation ignores FmFnN and GammaN
            rate_gain_parameters = rate_gain_parameters(1:5);
        end
        [gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total] = deal(rate_gain_parameters{:});
        FmFnN = []; % set these dummy variables to pass into the function below
        GammaN = [];
    else % multimode or higher-order modes
        [gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN] = deal(rate_gain_parameters{:});
    end
    % For fundamental mode, the computation of the gain amplification
    % factor is faster with CPU if the number of point < ~2^20.
    if sim.gpu_yes && ~isequal(rate_gain_parameters{1}.midx,1) % not fundamental mode
        %cross_sections_pump = structfun(@(c) gpuArray(c),cross_sections_pump,'UniformOutput',false);
        %overlap_factor = structfun(@(c) gpuArray(c),overlap_factor,'UniformOutput',false);
        cross_sections = structfun(@(c) gpuArray(c),cross_sections,'UniformOutput',false);
        overlap_factor.signal = gpuArray(overlap_factor.signal);
        N_total = gpuArray(N_total);
        FmFnN = gpuArray(FmFnN);
        GammaN = gpuArray(GammaN);
    end
    
    if gain_rate_eqn.export_N2
        [A_out,Power,...
         save_i,save_z,save_deltaZ,...
         t_delay_out,...
         N2]             = GMMNLSE_adaptive_rategain(sim,gain_rate_eqn,...
                                                     cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,...
                                                     save_z,save_deltaZ,save_points,t_delay_out,...
                                                     initial_condition,...
                                                     prefactor, omegas, D_op,...
                                                     SK_info, SRa_info, SRb_info, haw, hbw);
    else
        [A_out,Power,...
         save_i,save_z,save_deltaZ,...
         t_delay_out]    = GMMNLSE_adaptive_rategain(sim,gain_rate_eqn,...
                                                     cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN,...
                                                     save_z,save_deltaZ,save_points,t_delay_out,...
                                                     initial_condition,...
                                                     prefactor, omegas, D_op,...
                                                     SK_info, SRa_info, SRb_info, haw, hbw);
    end
% -------------------------------------------------------------------------
else % No gain, SM-gain, or new-gain, Taylor-gain model
    last_A = initial_condition.fields;
    
    % Create a progress bar first
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
        progress_bar_z = (1:num_progress_updates)*fiber.L0/num_progress_updates;
        progress_bar_i = 1;
    end

    % Then start the propagation
    z = 0;
    t_delay = 0; % time delay
    save_i = 2; % the 1st one is the initial field
    a5 = [];
    if ~isfield(sim,'max_deltaZ')
        sim.max_deltaZ = sim.save_period/10;
    end
    sim.deltaZ = sim.max_deltaZ;
    sim.last_deltaZ = 1;
    while z+eps(z) < fiber.L0 % eps(z) here is necessary due to the numerical error
        % Check for Cancel button press
        if sim.progress_bar && getappdata(h_progress_bar,'canceling')
            error('GMMNLSE_propagate:ProgressBarBreak',...
                  'The "cancel" button of the progress bar has been clicked.');
        end

        ever_fail = false;
        previous_A = last_A;
        previous_a5 = a5;

        success = false;
        while ~success
            if ever_fail
                last_A = previous_A;
                a5 = previous_a5;
            end

            switch sim.gain_model
                case 0
                    [last_A, a5,...
                     opt_deltaZ, success] = GMMNLSE_RK4IP_nogain_adaptive(    last_A, dt, sim, prefactor, SRa_info, SRb_info, SK_info, D_op, haw, hbw, a5);
                case 1
                    [last_A, a5,...
                     opt_deltaZ, success] = GMMNLSE_RK4IP_SMgain_adaptive(    last_A, dt, sim, prefactor, SRa_info, SRb_info, SK_info, D_op, haw, hbw, a5, G, fiber.saturation_energy);
                case 2
                    [last_A, a5,...
                     opt_deltaZ, success] = GMMNLSE_RK4IP_newgain_adaptive(   last_A, dt, sim, prefactor, SRa_info, SRb_info, SK_info, D_op, haw, hbw, a5, G, fiber.saturation_intensity, fiber.fr);
                case 3
                    [last_A,a5,...
                     opt_deltaZ, success] = GMMNLSE_RK4IP_taylorgain_adaptive(last_A, dt, sim, prefactor, SRa_info, SRb_info, SK_info, D_op, haw, hbw, a5, G, fiber.saturation_intensity, fiber.fr);
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
                if any(any(isnan(last_A))) %any(isnan(last_result),'all')
                    error('GMMNLSE_propagate:NaNError',...
                        'NaN field encountered, aborting.\nPossible reason is that the nonlinear length is too close to the large step size, deltaZ for split-step.');
                end
            end
        end
        
        sim.last_deltaZ = sim.deltaZ; % previous deltaZ

        % Center the pulse
        last_A_in_time = fft(last_A);
        tCenter = floor(sum(sum((-floor(Nt/2):floor((Nt-1)/2))'.*abs(last_A_in_time).^2),2)/sum(sum(abs(last_A_in_time).^2),2));
        % Because circshift is slow on GPU, I discard it.
        %last_result = ifft(circshift(last_A_in_time,-tCenter));
        if tCenter ~= 0
            if tCenter > 0
                last_A = ifft([last_A_in_time(1+tCenter:end,:);last_A_in_time(1:tCenter,:)]);
            elseif tCenter < 0
                last_A = ifft([last_A_in_time(end+1+tCenter:end,:);last_A_in_time(1:end+tCenter,:)]);
            end
            if sim.gpu_yes
                tCenter = gather(tCenter);
            end
            t_delay = t_delay + tCenter*initial_condition.dt;
        end

        % Update z
        z = z + sim.deltaZ;
        sim.deltaZ = min([opt_deltaZ,fiber.L0-z,sim.max_deltaZ]);

        % If it's time to save, get the result from the GPU if necessary,
        % transform to the time domain, and save it
        if z >= save_z(save_i)
            A_out_ii = fft(last_A);
            if sim.gpu_yes
                save_deltaZ(save_i-1) = gather(sim.last_deltaZ);
                save_z(save_i) = gather(z);
                A_out(:, :, save_i) = gather(A_out_ii);
            else
                save_deltaZ(save_i-1) = sim.last_deltaZ;
                save_z(save_i) = z;
                A_out(:, :, save_i) = A_out_ii;
            end

            t_delay_out(save_i) = t_delay;

            save_i = save_i + 1;
        end

        % Report current status in the progress bar's message field
        if sim.progress_bar
            if z >= progress_bar_z(progress_bar_i)
                waitbar(gather(z/fiber.L0),h_progress_bar,sprintf('%s%6.1f%%',sim.progress_bar_name,z/fiber.L0*100));
                progress_bar_i = find(z<progress_bar_z,1);
            end
        end
    end
end

% -------------------------------------------------------------------------

% Just to get an accurate timing, wait before recording the time
if sim.gpu_yes
    sim.betas = gather(sim.betas);
    wait(sim.gpuDevice.Device);
end
fulltime = toc(run_start);

%% Save the results in a struct

foutput.z = save_z(1:save_i-1);
foutput.deltaZ = save_deltaZ(1:save_i-2);
foutput.fields = A_out(:,:,1:save_i-1);
foutput.dt = initial_condition.dt;
foutput.betas = sim.betas;
foutput.seconds = fulltime;
foutput.t_delay = t_delay_out(1:save_i-1);
if sim.gain_model == 4
    foutput.Power = Power;
    if gain_rate_eqn.export_N2
        foutput.N2 = N2;
    end
end

end

%% Helper functions
% =========================================================================
function check_nummodes(sim, fiber, fields)
%CHECK_NUMMODES It checks the consistency of the number of modes of betas, SR tensors, and fields.

% Get the number of spatial modes from "SR".
num_spatial_modes = size(fiber.SR,1);

% -------------------------------------------------------------------------
% Check the number of spatial modes from betas and SR.
if sim.scalar
    num_spatial_modes_betas = size(fiber.betas,2);
else % polarized fields
    num_modes_betas = size(fiber.betas,2);
    if num_modes_betas == num_spatial_modes
        num_spatial_modes_betas = num_modes_betas;
    else % betas has already included polarization modes
        num_spatial_modes_betas = num_modes_betas/2;
    end
end

if num_spatial_modes_betas ~= num_spatial_modes
    if size(fiber.betas,1)==1 % if the user give betas in row vector for single mode. It should be in column vector.
        error('GMMNLSE_propagate:NumModesError',...
            '"betas" of each mode should be be in the form of "column vector".\nThe number of spatial modes of betas and SR tensors should be the same.');
    else
        error('GMMNLSE_propagate:NumModesError',...
            'The number of spatial modes of betas and SR tensors should be the same.');
    end
end

% -------------------------------------------------------------------------
% Check the number of modes of fields
num_modes_fields = size(fields,2); % spatial modes (+ polarization modes)

field_modes_mismatched = false;
if sim.scalar
    if num_modes_fields ~= num_spatial_modes
        field_modes_mismatched = true;
    end
else
    if num_modes_fields ~= 2*num_spatial_modes
        field_modes_mismatched = true;
    end
end
if field_modes_mismatched
    error('GMMNLSE_propagate:NumModesError',...
        'The number of modes of fields doesn''t match those of betas and SR tensors.\nIf not scalar fields, num_modes(field)=2*num_spatial_modes(SR,betas).');
end

end
% -------------------------------------------------------------------------
function fiber = betas_expansion_including_polarization_modes(sim,fiber,num_modes)
%BETAS_EXPANSION_INCLUDING_POLARIZATION_MODES It extends betas into 2*num_spatial_modes if necessary.

num_modes_betas = size(fiber.betas,2);

if ~sim.scalar
    if num_modes_betas == num_modes/2 % num_modes = 2*num_spatial_modes
        betas(:,2:2:num_modes) = fiber.betas;
        betas(:,1:2:num_modes-1) = fiber.betas;
        fiber.betas = betas;
    end
end

end
% -------------------------------------------------------------------------
function rev_idx = reverse_linear_indexing(idx,n)
%REVERSE_LINEAR_INDEXING
%
%   MATLAB linear indexing starts from the first dimension and so on, e.g.,
%   A = [7 5; is expanded into [7 8 1 9 5 2 4 1] while its indices correspond to [1 5;
%        8 2;                                                                     2 6;
%        1 4;                                                                     3 7;
%        9 1]                                                                     4 8]
%
%   Please refer to MATLAB linear indexing for details.
%
%   However, in SR and SK summation calculation, "Ra" and "nonlinear" loop  
%   over the first two indices. If "nonzero_midx" can be sorted in a way 
%   that doesn't move back and forth the memory all the time, it should be
%   faster...probably. So I add this function to correct the linear 
%   indexing direction given from "find" function.
%
%   Take the above as an example, the result indexing should now be [1 2;  while the first two rows correspond to midx1 and midx2.
%                                                                    3 4;
%                                                                    5 6;
%                                                                    7 8]
% -------------------------------------------------------------------------
% Original linear indexing direction:
%   nonzero_midx1234s = find(fiber.SR);
%
% Corrected linear indexing direction:
%   nonzero_midx1234s = find(permute(fiber.SR,[4 3 2 1]));
%   nonzero_midx1234s = reverse_linear_indexing(nonzero_midx1234s,num_spatial_modes);

rev_matrix = permute(reshape(1:n^4,n,n,n,n),[4 3 2 1]);
rev_idx = rev_matrix(idx);

end
% -------------------------------------------------------------------------
function DW = create_damped_window(Nt,lr_ratio,factor,min_dB)

t = (1:Nt)'/Nt;

offset = sqrt(1/(-log(10^(min_dB/10))))*factor;

if lr_ratio >= 0.1
    zero_ratio = lr_ratio - 0.1;
else
    zero_ratio = 0;
end
tzl = t(1:ceil(Nt*zero_ratio));
tl = t(ceil(Nt*zero_ratio)+1:ceil(Nt*lr_ratio)); tl = tl - tl(1) + offset;
tc = t(ceil(Nt*lr_ratio)+1:floor(Nt*(1-lr_ratio))-1);
tr = t(floor(Nt*(1-lr_ratio)):floor(Nt*(1-lr_ratio+0.1))); tr = tr - tr(end) - offset;
tzr = t(floor(Nt*(1-lr_ratio+0.1))+1:end);

ang = acos(exp(-1/(tl(end)/factor)^2));
DW = [0*10^(min_dB/10)*ones(length(tzl),1);exp(-1./(tl/factor).^2);cos(linspace(-ang,ang,length(tc))');exp(-1./(tr/factor).^2);0*10^(min_dB/10)*ones(length(tzr),1)];

end
% -------------------------------------------------------------------------
function cleanMeUp(h_progress_bar)
%CLEANMEUP It deletes the progress bar.

% DELETE the progress bar; don't try to CLOSE it.
delete(h_progress_bar);
    
end
% -------------------------------------------------------------------------