function beta = solve_for_EH_beta_func(wavelength,n_gas,sim,gas)
%   
%   wavelength: (m)
%   gas: a: core radius (m)

% The order of modes, based on propagation constant (beta), is found by
% running "find_order_of_EH_modes" first.

if ispc
    sep_char = '\';
else % unix
    sep_char = '/';
end

% Load the folder
current_path = mfilename('fullpath');
sep_pos = strfind(current_path,sep_char);
current_folder = current_path(1:sep_pos(end));
addpath([current_folder 'SLMtools/']);

% (n,m) modes to solve for
user_midx = [1,4,9,17,28,40]; % a maximum of 6 circular symmetric modes included here
user_midx = user_midx(sim.midx);
num_modes = length(user_midx);

% refractive index
load([current_folder sep_char 'n_silica.mat'],'slm');
n_out = slmeval(wavelength*1e6,slm,0); % refractive index of silica

%%
load([current_folder sep_char 'nm_order.mat'],'sorted_nm');
nm = sorted_nm(:,user_midx);
nm2 = zeros(size(nm));
for midx = 1:num_modes
    switch nm(1,midx)
        case 1
            nm2(:,midx) = nm(:,midx);
        case 2
            nm2(:,midx) = [0;nm(2,midx)];
        otherwise
            nm2(:,midx) = [2-nm(1,midx),nm(2,midx)];
    end
end
nm = [nm,nm2];

wavelength_sampling = length(wavelength);
k0 = 2*pi./wavelength;

vn = zeros(wavelength_sampling,num_modes*2);
unm = zeros(1,num_modes*2);
mode = cell(1,num_modes*2);
for midx = 1:num_modes*2
    % vn
    if nm(1,midx) == 0 % only TE is considered in this code
        mode{midx} = 'TE';
    else
        mode{midx} = 'EH';
    end
    
    vn(:,midx) = calc_vn(n_gas,n_out,mode{midx});
    
    % unm: zeros of the Bessel function of the first kind
    u = besselzero(nm(1,midx)-1,nm(2,midx),1);
    unm(midx) = u(end);
end

vn = gpuArray(vn);
k0 = gpuArray(k0);
unm = gpuArray(unm);
n_gas = gpuArray(n_gas);

ki = (1i*vn.*nm(1,:)./k0/gas.core_radius-1).*unm./(1i*vn./k0/gas.core_radius.*(nm(1,:)-1)-1)/gas.core_radius;
gamma = sqrt((k0.*n_gas).^2 - ki.^2);

chosen_midx = [];
next_midx = 1;
for midx = 1:num_modes
    if nm(1,midx) == 1
        acum_midx = next_midx+ [0,1];
    else
        acum_midx = next_midx + (0:3);
    end
    next_midx = next_midx + 4;
    chosen_midx = [chosen_midx,acum_midx];
end
beta = complex(zeros(wavelength_sampling,num_modes*4,'gpuArray'));
beta(:,1:4:end) = gamma(:,1:num_modes);
beta(:,2:4:end) = gamma(:,1:num_modes);
beta(:,3:4:end) = gamma(:,1:num_modes);
beta(:,4:4:end) = gamma(:,1:num_modes);
beta = beta(:,chosen_midx);
beta = gather(beta); % gather the output back from GPU

end