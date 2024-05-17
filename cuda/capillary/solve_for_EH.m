% The order of modes, based on propagation constant (beta), is found by
% running "find_order_of_EH_modes" first.

clearvars; close all;
addpath('SMLtools');

use_gpu = true; % GPU
pressure = 1.01325e5*1; % bar
temperature = 288.15; % 15 degree Celsius
core_radius = 12.5e-6; % core radius; m

% Don't change "wavelength"!
wavelength = [linspace(100,110,1000),linspace(110.01,160.01,100),linspace(160.51,2000,1000),linspace(2001.9,20000,1000)]'; % nm
target_wavelength = 800; % nm

% the number of sampling points of the spatial fields
r_sampling = 101;
theta_sampling = 101;

% (n,m) modes to solve for
user_midx = [1,4,9,17,28,40];
num_modes = length(user_midx);

% refractive index
% (n-1) is proportional to the pressure
% Reference: Walter G., et el, "On the Dependence of the Refractive Index of Gases on Temperature" (1903)
pressure0 = 1.01325e5; % Pa
temperature0 = 273.15; % 0 degree Celsius
refractivity = @(wavenumber) 14895.6./(180.7-wavenumber.^2) + 4903.7./(92-wavenumber.^2); % 10^6(n-1)
n_gas = refractivity(1./wavelength*1e3)/1e6*pressure/temperature/(pressure0/temperature0) + 1; % refractive index of H2
load('n_silica.mat','slm');
n_out = slmeval(wavelength/1e3,slm,0); % refractive index of silica

saved_filename = 'info_25um.mat';

%%
load('nm_order.mat');
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

wavelength = wavelength*1e-9; % m
target_wavelength = target_wavelength*1e-9; % m
if size(wavelength,1) == 1 % make it column vector
    wavelength = wavelength.';
end
wavelength_sampling = length(wavelength);
k0 = 2*pi./wavelength;

r = permute(linspace(core_radius/r_sampling*1e-3,core_radius,r_sampling),[1,3,2]);
theta = permute(linspace(0,2*pi,theta_sampling+1),[1,3,4,2]);  theta = theta(1:end-1); % 0 and 2*pi are the same
dr = core_radius/r_sampling;
dtheta = 2*pi/theta_sampling;

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
    wavelength = gpuArray(wavelength);
end

target_wavelength_sampling = length(target_wavelength);

ki = (1i*vn.*nm(1,:)./k0/core_radius-1).*unm./(1i*vn./k0/core_radius.*(nm(1,:)-1)-1)/core_radius;
gamma = sqrt((k0.*n_gas).^2 - ki.^2);

% interpolate to the target wavelength
if use_gpu
    if ispc
        sep_char = '\';
    else % unix
        sep_char = '/';
    end
    cuda_interp1_D2 = setup_kernel('interp1_D2','../cuda/',sep_char,target_wavelength_sampling*num_modes*2);
    target_ki = complex(zeros(target_wavelength_sampling,num_modes*2,'gpuArray'));
    target_ki = feval(cuda_interp1_D2,...
                                     wavelength,ki,false,uint32(wavelength_sampling),...
                                     target_wavelength,target_ki,false,uint32(target_wavelength_sampling),uint32(length(target_ki)),...
                                     0.5);
    target_k0 = complex(zeros(target_wavelength_sampling,1,'gpuArray'));
    target_k0 = feval(cuda_interp1_D2,...
                                     wavelength,ki,false,uint32(wavelength_sampling),...
                                     target_wavelength,target_k0,false,uint32(target_wavelength_sampling),uint32(length(target_k0)),...
                                     0.5);
else
    abs_ki = interp1(wavelength,abs(ki),target_wavelength);
    ang_ki = interp1(wavelength,unwrap(angle(ki),[],1),target_wavelength);
    target_ki = abs_ki.*exp(1i*ang_ki);
    
    abs_k0 = interp1(wavelength,abs(k0),target_wavelength);
    ang_k0 = interp1(wavelength,unwrap(angle(k0),[],1),target_wavelength);
    target_k0 = abs_k0.*exp(1i*ang_k0);
end
target_n_in = interp1(wavelength,n_gas,target_wavelength);
target_n_out = interp1(wavelength,n_out,target_wavelength);

num_polarized = 2; % two polarizations (r,theta)
mode_profiles = complex(zeros(target_wavelength_sampling,num_modes*2,r_sampling,theta_sampling,num_polarized));
% "GPU besselj" doesn't allow complex double" input, so I need to gather 
% the data back from GPU.
if use_gpu
    mode_profiles = gpuArray(mode_profiles);
    target_ki = gather(target_ki);
    target_k0 = gather(target_k0);
    target_n_in = gather(target_n_in);
    unm = gather(unm);
    theta = gather(theta);
end
for midx = 1:num_modes*2
    switch mode{midx}
        case 'TE'
            mode_profiles(:,midx,:,:,2) = repmat(besselj(1,target_ki(:,midx).*r),[1,1,1,theta_sampling,1]);
        case 'EH'
            theta0 = 0;
            Dbesselj = @(n,z) -besselj(n+1,z) + n./z.*besselj(n,z);
            mode_profiles(:,midx,:,:,1) = (besselj(nm(1,midx)-1,target_ki(:,midx).*r)+1i*unm(midx)./(2*target_k0.*target_n_in.*r).*sqrt((target_n_out./target_n_in).^2-1).*besselj(nm(1,midx),target_ki(:,midx).*r)).*sin(nm(1,midx)*(theta+theta0));
            mode_profiles(:,midx,:,:,2) = (besselj(nm(1,midx)-1,target_ki(:,midx).*r)+1i*unm(midx)^2./(2*nm(1,midx)*target_k0.*target_n_in*core_radius).*sqrt((target_n_out./target_n_in).^2-1).*Dbesselj(nm(1,midx),target_ki(:,midx).*r)).*cos(nm(1,midx)*(theta+theta0));
    end
end
norm = sqrt(sum(sum(sum(abs(mode_profiles).^2,5).*r*dr*dtheta,3),4));
norm = repmat(norm,[1,1,r_sampling,theta_sampling,num_polarized]);
nonzero_idx = norm~=0;
mode_profiles(nonzero_idx) = mode_profiles(nonzero_idx)./norm(nonzero_idx);

% Transform into (x,y) basis
if use_gpu
    mode_profiles_xy = zeros(size(mode_profiles),'gpuArray');
else
    mode_profiles_xy = zeros(size(mode_profiles));
end
mode_profiles_xy(:,:,:,:,1) = mode_profiles(:,:,:,:,1).*cos(theta) - mode_profiles(:,:,:,:,2).*sin(theta);
mode_profiles_xy(:,:,:,:,2) = mode_profiles(:,:,:,:,1).*sin(theta) + mode_profiles(:,:,:,:,2).*cos(theta);

%% Form a basis of linear polarizations
linear_mode_profiles = 1/sqrt(2)*(mode_profiles_xy(:,1:num_modes,:,:,:) + mode_profiles_xy(:,num_modes+1:num_modes*2,:,:,:));
linear_mode_profiles = cat(6,linear_mode_profiles,...
                       1/sqrt(2)*(mode_profiles_xy(:,1:num_modes,:,:,:) - mode_profiles_xy(:,num_modes+1:num_modes*2,:,:,:))...
                       ); % degenerate mode

% Because it may not align with x or y axis, I rotate the profiles.
center_wavelength_idx = ceil(target_wavelength_sampling/2);
xy_contribution = sum(sum(abs(linear_mode_profiles(center_wavelength_idx,:,:,:,:,:)).^2.*r*dr*dtheta,3),4);
xy_ratio = xy_contribution(:,:,:,:,1,:)./xy_contribution(:,:,:,:,2,:);
degenerate_mode_ratio = xy_ratio(:,:,:,:,:,1)./xy_ratio(:,:,:,:,:,2);
tol = 0.1;
rot = pi/4;
theta_shift = floor(rot/dtheta);
for midx = 1:num_modes
    if abs(degenerate_mode_ratio(:,midx,:,:) - 1) < tol
        linear_mode_profiles(:,midx,:,:,:,:) = rotate_mode_profiles(linear_mode_profiles(:,midx,:,:,:,:), rot, dtheta);
    end
end

% Now rotate it to have x-polarized modes before y-polarized ones
xy_contribution = sum(sum(abs(linear_mode_profiles(center_wavelength_idx,:,:,:,:,:)).^2.*r*dr*dtheta,3),4);
xy_ratio = xy_contribution(:,:,:,:,1,:)./xy_contribution(:,:,:,:,2,:);
degenerate_mode_ratio = xy_ratio(:,:,:,:,:,1)./xy_ratio(:,:,:,:,:,2);
for midx = 1:num_modes
    if degenerate_mode_ratio(:,midx) < 1
        linear_mode_profiles(:,midx,:,:,:,:) = rotate_mode_profiles(linear_mode_profiles(:,midx,:,:,:,:), pi/2, dtheta);
    elseif isnan(degenerate_mode_ratio(:,midx))
        if xy_contribution(:,midx,:,:,2,2) == 0
            if xy_ratio(:,midx,:,:,:,1) < 1 || isnan(xy_ratio(:,midx,:,:,:,1))
                linear_mode_profiles(:,midx,:,:,:,1) = rotate_mode_profiles(linear_mode_profiles(:,midx,:,:,:,1), pi/2, dtheta);
            end
        end
    end
end
num_orthogonal = 2;
if use_gpu
    all_modes_profiles = complex(zeros([size(linear_mode_profiles),num_orthogonal],'gpuArray'));
else
    all_modes_profiles = complex(zeros([size(linear_mode_profiles),num_orthogonal]));
end
all_modes_profiles(:,:,:,:,:,1,1) = linear_mode_profiles(:,:,:,:,:,1);
all_modes_profiles(:,:,:,:,:,2,2) = linear_mode_profiles(:,:,:,:,:,2);

all_modes_profiles(:,:,:,:,1,1,2) = -conj(linear_mode_profiles(:,:,:,:,2,1)); all_modes_profiles(:,:,:,:,2,1,2) = conj(linear_mode_profiles(:,:,:,:,1,1));
all_modes_profiles(:,:,:,:,1,2,1) = -conj(linear_mode_profiles(:,:,:,:,2,2)); all_modes_profiles(:,:,:,:,2,1,1) = conj(linear_mode_profiles(:,:,:,:,1,2));

%% 
mode_profiles = complex(zeros(target_wavelength_sampling,num_modes*4,r_sampling,theta_sampling,num_polarized));
if use_gpu
    mode_profiles = gpuArray(mode_profiles);
end
mode_profiles(:,1:4:end,:,:,:) = all_modes_profiles(:,:,:,:,:,1,1);
mode_profiles(:,2:4:end,:,:,:) = all_modes_profiles(:,:,:,:,:,1,2);
mode_profiles(:,3:4:end,:,:,:) = all_modes_profiles(:,:,:,:,:,2,1);
mode_profiles(:,4:4:end,:,:,:) = all_modes_profiles(:,:,:,:,:,2,2);

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
mode_profiles = mode_profiles(:,chosen_midx,:,:,:);

% Orthogonality
norm = sqrt(sum(sum(sum(abs(mode_profiles(:,1,:,:,:)).^2,5).*r*dr*dtheta,3),4));
mode_profiles(:,1,:,:,:) = mode_profiles(:,1,:,:,:)./norm;
for midx1 = 2:length(chosen_midx)
    for midx2 = 1:(midx1-1)
        mode_profiles(:,midx1,:,:,:) = mode_profiles(:,midx1,:,:,:) - sum(sum(sum(mode_profiles(:,midx1,:,:,:).*conj(mode_profiles(:,midx2,:,:,:)),5).*r*dr*dtheta,3),4).*mode_profiles(:,midx2,:,:,:);
    end
    norm = sqrt(sum(sum(sum(abs(mode_profiles(:,midx1,:,:,:)).^2,5).*r*dr*dtheta,3),4));
    mode_profiles(:,midx1,:,:,:) = mode_profiles(:,midx1,:,:,:)./norm;
end
%{
% Normalize it again
norm = sqrt(sum(sum(sum(abs(mode_profiles).^2,5).*r*dr*dtheta,3),4));
mode_profiles = mode_profiles./norm;
%}
%% Check orthogonality
midx1 = 1;
for midx2 = 1:length(chosen_midx)
    disp(sum(sum(sum(mode_profiles(ceil(target_wavelength_sampling/2),midx1,:,:,:).*conj(mode_profiles(ceil(target_wavelength_sampling/2),midx2,:,:,:)),5).*r*dr*dtheta,3),4));
end

%% Plot it!
x = squeeze(r.*cos(theta));
y = squeeze(r.*sin(theta));
for midx = 1:length(chosen_midx)
    %figure; polarPcolor(squeeze(r)',squeeze(theta)'*180/pi,squeeze(real(mode_profiles(ceil(wavelength_sampling/2),midx,:,:,1))));
    %figure; polarPcolor(squeeze(r)',squeeze(theta)'*180/pi,squeeze(real(mode_profiles(ceil(wavelength_sampling/2),midx,:,:,2))));
    
    figure;
    quiver(x,y,squeeze(real(mode_profiles(ceil(target_wavelength_sampling/2),midx,:,:,1))),squeeze(real(mode_profiles(ceil(target_wavelength_sampling/2),midx,:,:,2))));
    xlim([min(x(:)),max(x(:))]); ylim([min(y(:)),max(y(:))]);
end

%% Save data
r = squeeze(r);
theta = squeeze(theta);

% change the notation into "beta"
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
    wavelength = gather(wavelength);
    beta = gather(beta);
    mode_profiles = gather(mode_profiles);
end

save(saved_filename,'wavelength','beta','mode_profiles','r','theta');

%% SETUP_KERNEL
% -------------------------------------------------------------------------
    % ---------------------------------------------------------------------
    function recompile_ptx(cuda_dir_path,sep_char,cudaFilename,ptxFilename)
        if ispc
            system(['nvcc -ptx "', cuda_dir_path sep_char cudaFilename, '" --output-file "', cuda_dir_path sep_char ptxFilename '"']);
        else % unix
            % tested: Debian 9 (Stretch)
            % Cuda 8 doesn't support gcc6, beware to use gcc5 or clang-3.8.
            system(['nvcc -ccbin clang-3.8 -ptx "', cuda_dir_path sep_char cudaFilename, '" --output-file "', cuda_dir_path sep_char ptxFilename '"']);
        end
    end
    % ---------------------------------------------------------------------
function kernel = setup_kernel(filename,cuda_dir_path,sep_char,total_num)

cudaFilename = [filename, '.cu'];
ptxFilename = [filename, '.ptx'];

if ~exist([cuda_dir_path sep_char ptxFilename], 'file')
    recompile_ptx(cuda_dir_path,sep_char,cudaFilename,ptxFilename);
end

% Setup the kernel from the cu and ptx files
try
    kernel = parallel.gpu.CUDAKernel([cuda_dir_path sep_char ptxFilename], [cuda_dir_path sep_char cudaFilename]);
catch
    % Compile the CUDA code again.
    % Currently found error:
    %    version mismatch due to different versions of cuda I use in Windows and Debian.
    recompile_ptx(cuda_dir_path,sep_char,cudaFilename,ptxFilename);
    kernel = parallel.gpu.CUDAKernel([cuda_dir_path sep_char ptxFilename], [cuda_dir_path sep_char cudaFilename]);
end

if total_num > kernel.MaxThreadsPerBlock
    ThreadsPerBlock = kernel.MaxThreadsPerBlock;
else
    ThreadsPerBlock = total_num;
end
kernel.ThreadBlockSize = [ThreadsPerBlock,1,1];
kernel.GridSize =[ceil(total_num/ThreadsPerBlock), 1];

end