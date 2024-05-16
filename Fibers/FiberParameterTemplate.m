% created by BarakM 11.5.24
% here you set all the fiber parameters that you want to create.
% then use this function on "Create_Fiber" scrip

%% PLEASE
% make the folder and the function in the same name
%% Fiber parameters
% enter here all the relevant parameter for the fiber

function data = FiberParameterTemplate()
data.folder_name = 'Barak_GIF625_3modes_CW1550nm';    % folder where the output will be stored

% wavelength and modes
data.lambda0 = 1550e-9;                          % center wavelength [m]
data.lrange = 100e-9;                            % wavelength range [m]. If 0 only the center wavelength will be used
data.Nf = 100;                                     % number of frequency points at which the modes will be calculated
data.num_modes = 4;                              % number of modes to calculate

% fiber size and grid
data.radius = 62.5/2;                                 % outer radius of fiber [um]. 
data.spatial_window = data.radius * 3;                        % full spatial window size [um]
data.Nx = 512;                                  % number of spatial grid points


% optical fiber features
data.extra_params.ncore_diff = 0.0298;           % difference between the index at the center of the core, and the cl
data.extra_params.alpha = 2.00;                  % Shape parameter
data.extra_params.NA = 0.275;                     % Numerical Aperture

% fiber refractive index type
data.profile_function = @build_GRIN;             % function that builds the fiber ; build_GRIN\build_step

% dispersion polinomial fit
data.dispersion.polynomial_fit_order = 8;                    % polyfit order
data.dispersion.num_disp_orders = 2;                         % i.e. if this is 3, 4 coefficients will be calculated, including the 0th order

% SRSK tenson parameters
data.tenson_calc.linear_yes = 1;                         % 1 = linear polarization, 0 = circular polarization
data.tenson_calc.gpu_yes = 1;                            % 1 = run on GPU, 0 = run on CPU
data.tenson_calc.single_yes = 0;                         % 1 = single precision, 0 = double precision

mkdir(data.folder_name);
save([data.folder_name '/Fiber_params'], "data")

end