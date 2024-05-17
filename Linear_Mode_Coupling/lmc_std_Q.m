function std_Q = lmc_std_Q( sim,fiber,num_modes )
%LMC_STD_Q It calculates the coupling strength between or inside mode 
%groups, which will be multiplied with the corresponding random matrix with
%a standard deviation 1.
%
%   num_modes: the number of modes including polarization modes
%
%   std_Q: a matrix of the size (num_modes, num_modes); coupling strength

num_spatial_modes = num_modes/2;

% Extend "betas" into 2*num_spatial_modes if necessary.
fiber = betas_expansion_including_polarization_modes(fiber,num_modes);

sim.lmc.mode_groups = sim.lmc.mode_groups*2; % add polarization modes
num_mode_groups = length(sim.lmc.mode_groups);

% Generate the matrix of coupling lengths
tmp = mat2cell(ones(num_modes,num_modes),sim.lmc.mode_groups,sim.lmc.mode_groups);
for i = 1:num_mode_groups^2
    tmp{i} = tmp{i}*i;
end
Lcoup_idx = cell2mat(tmp);
sim.lmc.Lcoup = sim.lmc.Lcoup_spatialmode(Lcoup_idx);
tmp = mat2cell(sim.lmc.Lcoup_polarizedmode*ones(2,2,num_spatial_modes),2,2,ones(1,num_spatial_modes));
sim.lmc.Lcoup_polarizedmode = blkdiag(tmp{:});
sim.lmc.Lcoup(sim.lmc.Lcoup_polarizedmode~=0) = sim.lmc.Lcoup_polarizedmode(sim.lmc.Lcoup_polarizedmode~=0);

delta_beta0 = fiber.betas(1,:) - fiber.betas(1,:).';
var_Q = (pi./sim.lmc.Lcoup).^2 - (delta_beta0/2).^2;

% The diagonal part should be zero because this is the cross-coupling
% matrix
diag_idx = (1:num_modes) + (0:num_modes-1)*num_modes;
var_Q(diag_idx) = 0;

% The coupling between polarization modes should be determined only by
% Lcoup_polarizedmode, so I kill the coupling between different spatial
% modes with different polarizations here.
p1_idx = zeros(num_modes,1); p1_idx(1:2:end) = 1;
p2_idx = zeros(num_modes,1); p2_idx(2:2:end) = 1;
coupling_idx = p1_idx*p1_idx' + p2_idx*p2_idx'; % spatial-mode coupling
coupling_idx(diag_idx) = 0;
for midx = 1:2:num_modes % add polarization-mode coupling
    coupling_idx(midx,midx+1) = 1;
    coupling_idx(midx+1,midx) = 1;
end
var_Q = var_Q.*coupling_idx;

std_Q = sqrt(var_Q);

% Check the validity of coupling strength
% The coupling strength has a maximum value due to the existence of a
% nonzero delta_beta0. If delta_beta0=0 (degenerate modes), there's no
% maximum.
if any(var_Q(:)<0)
    max_Lcoup = abs(2*pi./delta_beta0);
    
    error('lmd_std_Q:QError',...
        ['The coupling length is too large. Its maximum is'...
            sprintf(['\n [ ' repmat('%10.4e ',1,num_modes) ']'],max_Lcoup.') '\n\n'... % show maximum coupling strength available
         'Current mode-group coupling strength matrix (a value multiplied to elements of a random matrix with std 1) is'...
            sprintf(['\n [ ' repmat('%10.4e ',1,num_modes) ']'],std_Q.') '\n\n'... % show current coupling strength
          '(NaN stands for imaginary coupling strength.)']);
end

end

%%
function fiber = betas_expansion_including_polarization_modes(fiber,num_modes)
%BETAS_EXPANSION_INCLUDING_POLARIZATION_MODES It extends betas into 2*num_spatial_modes if necessary.

num_modes_betas = size(fiber.betas,2);

if num_modes_betas == num_modes/2 % num_modes = 2*num_spatial_modes
    betas(:,2:2:num_modes) = fiber.betas;
    betas(:,1:2:num_modes-1) = fiber.betas;
    fiber.betas = betas;
elseif num_modes_betas ~= num_modes
    error('lmd_std_Q:NUM_MODESError',...
        'The number of modes of "betas" should be either "num_spatial_modes" or "2*num_spatial_modes".');
end

end