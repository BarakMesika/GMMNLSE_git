%   wavelength: (nm)
%   gas: a: core radius (m)

% The order of modes, based on propagation constant (beta), is found by
% running "find_order_of_EH_modes" first.

addpath('SLMtools');

pressure = 1.01325e5*5; % bar
temperature = 288.15; % 15 degree Celsius
core_radius = 12.5e-6; % m

% Don't change "wavelength"!
wavelength = [linspace(100,110,1000),linspace(110.01,160.01,100),linspace(160.51,2000,1000),linspace(2001.9,20000,1000)]'*1e-9; % m

if ispc
    sep_char = '\';
else % unix
    sep_char = '/';
end

use_gpu = true;

% (n,m) modes to solve for
user_midx = [1,4,9,17,28,40]; % a maximum of 6 circular symmetric modes included here
num_modes = length(user_midx);

% refractive index
% Reference:
% 1. Walter G., et el, "On the Dependence of the Refractive Index of Gases on Temperature" (1903)
% 2. Arthur L. Ruoff and Kouros Ghandehari, "THE REFRACTIVE INDEX OF HYDROGEN AS A FUNCTION OF PRESSURE" (1993)
pressure0 = 1.01325e5; % Pa
temperature0 = 273.15; % 0 degree Celsius
refractivity = @(wavenumber) 14895.6./(180.7-wavenumber.^2) + 4903.7./(92-wavenumber.^2); % 10^6(n-1)
n_gas = refractivity(1./wavelength*1e-6)/1e6*pressure/temperature/(pressure0/temperature0) + 1;
load('n_silica.mat','slm');
n_out = slmeval(wavelength*1e6,slm,0); % refractive index of silica

saved_filename = 'info_25um.mat';

%%
load('nm_order.mat','sorted_nm');
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

if use_gpu
    vn = gpuArray(vn);
    k0 = gpuArray(k0);
    unm = gpuArray(unm);
    n_gas = gpuArray(n_gas);
end

ki = (1i*vn.*nm(1,:)./k0/core_radius-1).*unm./(1i*vn./k0/core_radius.*(nm(1,:)-1)-1)/core_radius;
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
if use_gpu
    beta = complex(zeros(wavelength_sampling,num_modes*4,'gpuArray'));
else
    beta = complex(zeros(wavelength_sampling,num_modes*4));
end
beta(:,1:4:end) = gamma(:,1:num_modes);
beta(:,2:4:end) = gamma(:,1:num_modes);
beta(:,3:4:end) = gamma(:,1:num_modes);
beta(:,4:4:end) = gamma(:,1:num_modes);
beta = beta(:,chosen_midx);

if use_gpu
    beta = gather(beta);
end

save(saved_filename,'wavelength','beta');