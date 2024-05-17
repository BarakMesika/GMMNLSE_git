function plot_std_Q_vs_Lcoup(beta0,num_modes)
%PLOT_STD_Q_VS_LCOUP It computes the relationship between the coupling
%length and the coupling strength.
%   
%   beta0: the beta0 of a mode group; (1,num_spatial_modes) or (1,num_spatial_modes*2)
%   num_modes: the number of modes

% Extend "betas" into 2*num_spatial_modes if necessary.
beta0 = betas_expansion_including_polarization_modes(beta0,num_modes);

avg_beta0 = abs(mean(beta0));

% There are two ways to calculate delta_beta0:
%   1. 2 modes: their difference of beta0
%   2. many modes: the standard deviation of their beta0
if num_modes == 2
    delta_beta0 = diff(beta0);
else % more than 2 modes: delta_beta0 is taken from the standard deviation of beta0 in each mode group.
    delta_beta0 = std(beta0);
end

if delta_beta0 == 0
    Lcoup = 10.^(linspace(-3,5,1001));
else
    if num_modes == 2
        max_Lcoup = 2*pi./delta_beta0;
    else % assume no correlation: <beta_i*beta_j>=0
        max_Lcoup = sqrt(2)*pi./delta_beta0;
    end
    min_Lcoup = max_Lcoup/1e6; % the minimum Lcoup to plot for
    Lcoup = 10.^(linspace(log10(min_Lcoup),log10(max_Lcoup),1001));
    Lcoup = Lcoup(1:end-1); % the last point can be larger than max_Lcoup because of numerical inaccuracy, get rid of it
end

if num_modes == 2
    delta_beta_term = (delta_beta0/2)^2;
else % many modes
    delta_beta_term = (delta_beta0)^2/2;
end

var_Q = (pi./Lcoup).^2 - delta_beta_term;

figure('Name','Coupling strength');
loglog(Lcoup,sqrt(var_Q),Lcoup,ones(size(Lcoup))*avg_beta0);
xlabel('Coupling length (m)');
ylabel('Coupling strength');
if avg_beta0 ~= 0
    l = legend('Coupling strength(std(Q))','avg(\beta_0)');
else
    l = legend('Coupling strength(std(Q))');
end
set(l,'location','southwest');

intersection = InterX([log10(Lcoup); log10(sqrt(var_Q))],[log10(Lcoup); ones(size(Lcoup))*log10(avg_beta0)]);

if delta_beta0 ~= 0
    fprintf('Maximum Lcoup (beat length) = %2.4f(m)\n',max_Lcoup);
else
    disp('delta_beta0 = 0');
end
if avg_beta0 ~= 0
    if ~isempty(intersection)
        fprintf('Intersection at the coupling length = %4.4g(m)\n',10^intersection(1));
    end
else
    disp('avg(beta_0) = 0');
end

end

%%
function expanded_betas = betas_expansion_including_polarization_modes(betas,num_modes)
%BETAS_EXPANSION_INCLUDING_POLARIZATION_MODES It extends betas into 2*num_spatial_modes if necessary.

expanded_betas = betas;

num_modes_betas = size(betas,2);

if num_modes_betas == num_modes/2 % num_modes = 2*num_spatial_modes
    expanded_betas(:,2:2:num_modes) = betas;
    expanded_betas(:,1:2:num_modes-1) = betas;
elseif num_modes_betas ~= num_modes
    error('lmd_std_Q:NUM_MODESError',...
        'The number of modes of "betas" should be either "num_spatial_modes" or "2*num_spatial_modes".');
end

end