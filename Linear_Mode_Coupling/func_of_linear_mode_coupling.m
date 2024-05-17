function func = func_of_linear_mode_coupling
%FUNC_OF_LINEAR_MODE_COUPLING It includes several functions related to
%linear mode coupling used in "GMMNLSE_propagate.m"

func.lmc_check = @lmc_check; % check the random mode coupling input

func.lmc_Manakov_kappa = @lmc_Manakov_kappa; % calculate Manakov nonlinear coefficients

func.lmc_dispersion_operator = @lmc_dispersion_operator; % calculate the dispersion operator for linear mode coupling

func.lmc_parameters = @lmc_parameters; % calculate some parameters related to linear mode coupling.

end

%%
function lmc_check(sim)
%LMC_CHECK Check the random mode coupling inputs

switch sim.lmc.model
    case 1
        if ~isfield(sim.lmc,'Lcorr') || isempty(sim.lmc.Lcorr)
            error('GMMNLSE_propagate:LcorrError',...
                '"Lcorr" is necessary for "lmc_model 1 (R model)".');
        end
    case 2
        % Check the random mode coupling inputs
        if ~isfield(sim.lmc,'Lcorr') || isempty(sim.lmc.Lcorr)
            error('GMMNLSE_propagate:lmc_lengthError',...
                '"Lcorr" is necessary for "lmc_model 2 (iQA model)".');
        end
        if ~isfield(sim.lmc,'Lcoup_spatialmode') || isempty(sim.lmc.Lcoup_spatialmode)
            error('GMMNLSE_propagate:lmc_lengthError',...
                '"Lcoup_spatialmode" matrix for lmc_model=2 (iQA model) is required.');
        else
            num_mode_groups = length(sim.lmc.mode_groups);
            if ~isequal(size(sim.lmc.Lcoup_spatialmode),[num_mode_groups,num_mode_groups])
                error('GMMNLSE_propagate:lmc_lengthError',...
                    'The size of "sim.lmc.Lcoup_spatialmode" should be [num_mode_groups,num_mode_groups].');
            end
            if ~issymmetric(sim.lmc.Lcoup_spatialmode)
                error('GMMNLSE_propagate:lmc_lengthError',...
                    '"sim.lmc.Lcoup_spatialmode" needs to be symmetric.');
            end
        end
end

end

%%
function [kappaK,kappaR1,kappaR2] = lmc_Manakov_kappa( sim,num_modes,mode_info,SK_info )
%LMC_MANAKOV_KAPPA Calculate Manakov nonlinear coefficients

% Manakov equation always considers polarizations, which affects SRSK
% and the following nonlinear coefficients, kappaK, kappaR1, kappaR2.
if sim.scalar
    mode_groups_all = 2*sim.lmc.mode_groups;
    num_modes_all = num_modes*2;
else
    mode_groups_all = sim.lmc.mode_groups;
    num_modes_all = num_modes;
end
mode_group_idx_all = mat2cell(1:num_modes_all,1,mode_groups_all); % the index of modes in each mode group including polarization modes if running under polarized fields

% Recover SR,SK back to 4D array for "Manakov_SRSK"
if sim.gpu_yes
    if sim.single_yes
        SR = zeros(num_modes_all,num_modes_all,num_modes_all,num_modes_all,'single','gpuArray');
        SK = zeros(num_modes_all,num_modes_all,num_modes_all,num_modes_all,'single','gpuArray');
    else
        SR = zeros(num_modes_all,num_modes_all,num_modes_all,num_modes_all,'gpuArray');
        SK = zeros(num_modes_all,num_modes_all,num_modes_all,num_modes_all,'gpuArray');
    end
else
    if sim.single_yes
        SR = zeros(num_modes_all,num_modes_all,num_modes_all,num_modes_all,'single');
        SK = zeros(num_modes_all,num_modes_all,num_modes_all,num_modes_all,'single');
    else
        SR = zeros(num_modes_all,num_modes_all,num_modes_all,num_modes_all);
        SK = zeros(num_modes_all,num_modes_all,num_modes_all,num_modes_all);
    end
end
% SR
SRidx = sub2ind(num_modes_all*ones(1,4),mode_info.nonzero_midx1234s(1,:),mode_info.nonzero_midx1234s(2,:),mode_info.nonzero_midx1234s(3,:),mode_info.nonzero_midx1234s(4,:));
SR(SRidx') = mode_info.SRa;
% SK
SKidx = sub2ind(num_modes_all*ones(1,4),SK_info.nonzero_midx1234s(1,:),SK_info.nonzero_midx1234s(2,:),SK_info.nonzero_midx1234s(3,:),SK_info.nonzero_midx1234s(4,:));
SK(SKidx') = SK_info.SK;

% Compute the Manakov nonlinear coefficients, kappaK, kappaR1, kappaR2.
[kappaK,kappaR1,kappaR2] = Manakov_SRSK(SR,SK,mode_group_idx_all,sim);

end

%%
function D0_op = lmc_dispersion_operator( sim,D0_op,num_modes,N,mode_groups )
%LMC_DISPERSION_OPERATOR Calculate the dispersion operator for linear mode
%coupling
%
% If using lmc_model 2, it's necessary to arrange D0_op into a 
% (num_modes,num_modes,N) multidimensional array with a diagonal matrix in 
% each page.
% If using lmc_model 3, dispersion term is averaged in each mode group.

switch sim.lmc.model
    case 2
        D0_op_tmp = D0_op;
        if sim.gpu_yes
            if sim.single_yes
                D0_op = zeros(num_modes,num_modes,N,'single','gpuArray');
            else
                D0_op = zeros(num_modes,num_modes,N,'gpuArray');
            end
        else
            if sim.single_yes
                D0_op = zeros(num_modes,num_modes,N,'single');
            else
                D0_op = zeros(num_modes,num_modes,N);
            end
        end
        D0_op_diag_idx = bsxfun(@plus, ( 0:num_modes^2:(num_modes^2*(N-1)) )', ( 1:(num_modes+1):num_modes^2 ) ); % calculate the linear indices for the diagonal elements
        D0_op(D0_op_diag_idx(:)) = D0_op_tmp(:); % put D0_op_tmp into the diagonal elements of D0_op
    case 3 % Manakov equation
        if sim.gpu_yes % "mg" stands for "mode group"
            if sim.single_yes
                D0_op_mg = zeros(N,num_modes,'single','gpuArray');
            else
                D0_op_mg = zeros(N,num_modes,'gpuArray');
            end
        else
            if sim.single_yes
                D0_op_mg = zeros(N,num_modes,'single');
            else
                D0_op_mg = zeros(N,num_modes);
            end
        end
        for mg = 1:length(mode_groups)
            D0_op_mg(:,mode_groups{mg}) = bsxfun(@times, mean(D0_op(:,mode_groups{mg}),2),ones(1,sim.lmc.mode_groups(mg)));
        end
        D0_op = D0_op_mg;
end

end

%%
function [Lcorr_zpoints,num_rand_matrices_in_a_single_part,...
    lmc_idx_in_single_part,lmc_idx_over_parts,...
    RQ_matrices,rand_matrices_over_entire_L0] = lmc_parameters(fiber,sim,large_step,z_points,num_modes,std_Q)
%CALC_RMC_PARAMETERS It calculates some parameters related to linear mode coupling.

% Create a dummy variable to pass through the function below for R model.
if sim.lmc.model==1
    std_Q = [];
end

% Find the z points of each correlation length.
num_RQ = ceil(fiber.L0/sim.lmc.Lcorr); % the number of Lcorr within L0
if num_RQ >= z_points-1 % Lcorr <= large_step
    num_RQ = z_points-1;
    Lcorr_zpoints = 2:z_points;
    lmc_idx_in_single_part = 0; % in this case, num_RQ = num(Lcorr_points)
elseif ismember(num_RQ,[0,1]) % Lcorr >= fiber.L0 or Lcorr = inf
    num_RQ = 1;
    Lcorr_zpoints = [];
    lmc_idx_in_single_part = [];
else
    Lcorr_zpoints = ceil(sim.lmc.Lcorr*(1:num_RQ-1)/large_step)+1; % the points corresponding to the multiples of Lcorr
    lmc_idx_in_single_part = 1; % in this case, num_RQ = num(Lcorr_points)-1
end

% Generate the random matrices and separate them in a few parts for the
% purpose of sending them in GPU one by one if it's in use to avoid
% blowing up the GPU momery.
if sim.single_yes
    running_precision = 4;
else
    running_precision = 8;
end
max_memory = 100*2^(20); % set maximum memory, 100MB, to avoid blowing up GPU memory
num_rand_matrices_in_a_single_part = floor(max_memory/(running_precision*2*num_modes^2)/3); % the number of matrices in a single part
rand_matrices_over_entire_L0 = {};
num_parts = ceil(num_RQ/num_rand_matrices_in_a_single_part); % the total number of parts
each_num_RQ = [num_rand_matrices_in_a_single_part*ones(1,num_parts-1) 0];
if rem(num_RQ,num_rand_matrices_in_a_single_part) == 0
    each_num_RQ(end) = num_rand_matrices_in_a_single_part;
else
    each_num_RQ(end) = rem(num_RQ,num_rand_matrices_in_a_single_part);
end

% Since "blkdiag_rand" will create matrices in GPU directly if GPU is
% used, I first create the first part of the random matrices in GPU, 
% then generate the other parts in the the RAM.
if sim.gpu_yes
    % Create the first part of random matrices in GPU.
    rand_func1 = blkdiag_rand(sim);
    RQ_matrices = create_rand_matrices_for_RQ(sim,num_modes,rand_func1,each_num_RQ(1),std_Q);

    start_idx_rand_mat_in_memory = 2; % the 1st part has already been created, start with the 2nd part of random-matrix generation below
else
    start_idx_rand_mat_in_memory = 1; % if not using GPU, there's no need to create the 1st part of random matrices in advanced
end

if length(each_num_RQ) > 1 || ~sim.gpu_yes % "we need more random matrices" or "we're using CPU (the 1st part of random matrices hasn't been created yet)"
    % Create (the rest of) random matrices in RAM.
    rand_func = blkdiag_rand(struct('single_yes',sim.single_yes,'gpu_yes',false));
    num_create_RQ = num_RQ-each_num_RQ(1)*(start_idx_rand_mat_in_memory==2);
    rand_matrices_tmp = create_rand_matrices_for_RQ(sim,num_modes,rand_func,num_create_RQ,std_Q);
    rand_matrices_over_entire_L0 = mat2cell(rand_matrices_tmp,...
                                    num_modes,num_modes,each_num_RQ(start_idx_rand_mat_in_memory:end) );
    if ~sim.gpu_yes % CPU
        RQ_matrices = rand_matrices_over_entire_L0{1};
        lmc_idx_over_parts = 1; % the starting loop-index(+1) for parts of random matrices
                                  % the 1st part has already been stored in "all_RQ" in the lines above, so it starts with 2 in the propagation
    else % GPU
        lmc_idx_over_parts = 0; % the starting loop-index(+1) for parts of random matrices
                                  % since the 1st part of random matrices has already stored in GPU, it starts with 1 in the propagation
    end
else
    lmc_idx_over_parts = []; % There's only one part of random matrices
end

end

%%
function RQ_matrices = create_rand_matrices_for_RQ(sim,num_modes,rand_func,num_RQ,std_Q)
%CREATE_RAND_MATRICES_FOR_RQ As the name suggests, it creates R or Q
%matrices. It's a sub-function of "lmc_parameters" above.

switch sim.lmc.model
    case 1
        % Create unitary matrices
        RQ_matrices = rand_func.haar(sim.lmc.mode_groups*2,num_RQ);
    case 2
        % Create hermitian matrices
        RQ_matrices = rand_func.hermitian(num_modes,num_RQ);
        RQ_matrices = RQ_matrices.*std_Q;
end

end