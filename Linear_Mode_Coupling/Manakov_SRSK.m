function [kappaK,kappaR1,kappaR2] = Manakov_SRSK(SR,SK,mode_groups,sim)
%MANAKOV_SRSK It calculates the nonlinear coefficients of Manakov
%equations.
%
%   SR and SK need to include both polarization and spatial modes for
%   Manakov equations to be correct. Besides, polarization modes contribute
%   the most in the coupling because of their degenerate feature in a
%   typical amorphous SMF.
%   
%   mode_groups is a cell in which each element contains the mode indices
%   of modes in each mode group. For example, mode_groups = {[1 2],[3 4 5]}
%   represents that mode group 1 has mode 1 and 2, while mode group 2 has
%   mode 3, 4, and 5.
%
%   sim is a structure having three fields:
%       scalar, gain_model, single_yes
%
%   The exact formulae of these nonlinear coefficients are in my writeup,
%   "Random Mode Coupling" by Yi-Hao Chen, Ph.D. in AEP, Cornell University.

N = cellfun(@length,mode_groups); % the number of modes in each mode group
NlNm = N'*N;

num_mode_groups = length(mode_groups);
diag_idx = 1:(num_mode_groups+1):num_mode_groups^2;

num_modes = sum(N);

%% Kerr
NK = NlNm;
NK(diag_idx) = N.*(N+1);
[SK_llmm,SK_lmlm] = calc_Suv(SK,mode_groups,num_modes,sim);
kappaK = (SK_llmm+SK_lmlm)./NK;

%% Raman
[SR_llmm,SR_lmlm] = calc_Suv(SR,mode_groups,num_modes,sim);
kappaR1 = SR_llmm./NlNm;
kappaR2 = SR_lmlm./NlNm;
kappaR1(diag_idx) = SR_llmm(diag_idx)./(N.^2-1)-SR_lmlm(diag_idx)./N./(N.^2-1);
kappaR2(diag_idx) = SR_lmlm(diag_idx)./(N.^2-1)-SR_llmm(diag_idx)./N./(N.^2-1);

% Ni=1 mode group:
% If Ni=1, the factors calculated above give "NaN". It's updated below.
m1 = find(N==1);
if ~isempty(m1)
    m1_diag_idx = (m1-1)*num_mode_groups+m1;
    kappaR1(m1_diag_idx) = SR_llmm(m1_diag_idx)/2;
    kappaR2(m1_diag_idx) = SR_lmlm(m1_diag_idx)/2;
end

%% Expand it into the form convenient for the multiplication later
% The calculation of kappaK, kappaR1, kappaR2 terms need to consider
% polarizations, but for the computation later, if under scalar fields,
% a factor of 2 for the number of modes is reduced.
if sim.scalar
    exclude_polarization = 2;
    num_modes = num_modes/2;
else
    exclude_polarization = 1;
end
expand_idx = zeros(1,num_modes);
previous_i = [0 cumsum(N/exclude_polarization)];
for mg = 1:num_mode_groups
    i = previous_i(mg);
    expand_idx((1:N(mg)/exclude_polarization)+i) = mg*ones(1,N(mg)/exclude_polarization);
end
expand_fun = @(ka) ka(expand_idx,:);

kappaK = permute(expand_fun(kappaK),[3 1 4 2]);
if sim.gain_model ~= 2
    kappaR1 = permute(expand_fun(kappaR1),[3 1 4 2]);
    kappaR2 = permute(expand_fun(kappaR2),[3 1 4 2]);
else
    % For "Manakov new-gain model", we need to compute the transfer matrix as well.
    kappaR1 = struct('T',kappaR1                       ,'NL',permute(expand_fun(kappaR1),[3 1 4 2]));
    kappaR2 = struct('T',kappaR2(expand_idx,expand_idx),'NL',permute(expand_fun(kappaR2),[3 1 4 2]));
end

end

%% calc_Suv
function [S_llmm,S_lmlm] = calc_Suv(S,mode_groups,num_modes,sim)
%CALC_SUV It calculates S_llmm and S_lmlm terms

num_mode_groups = length(mode_groups);

if sim.gpu_yes
    if sim.single_yes
        S_llmm = zeros(num_mode_groups,num_mode_groups,'single','gpuArray');
        S_lmlm = zeros(num_mode_groups,num_mode_groups,'single','gpuArray');
    else
        S_llmm = zeros(num_mode_groups,num_mode_groups,'gpuArray');
        S_lmlm = zeros(num_mode_groups,num_mode_groups,'gpuArray');
    end
else
    if sim.single_yes
        S_llmm = zeros(num_mode_groups,num_mode_groups,'single');
        S_lmlm = zeros(num_mode_groups,num_mode_groups,'single');
    else
        S_llmm = zeros(num_mode_groups,num_mode_groups);
        S_lmlm = zeros(num_mode_groups,num_mode_groups);
    end
end
for i = 1:num_mode_groups
     for j = 1:num_mode_groups
         u = mode_groups{i};
         v = mode_groups{j};
         l = repmat(u,1,length(v));
         m = kron(v,ones(1,length(u)));
         llmm = [l;l;m;m]';
         lmlm = [l;m;l;m]';
         llmm_idx = sub2ind(num_modes*ones(1,4),llmm(:,1),llmm(:,2),llmm(:,3),llmm(:,4));
         lmlm_idx = sub2ind(num_modes*ones(1,4),lmlm(:,1),lmlm(:,2),lmlm(:,3),lmlm(:,4));
         S_llmm(i,j) = sum(S(llmm_idx));
         S_lmlm(i,j) = sum(S(lmlm_idx));
     end
end

end