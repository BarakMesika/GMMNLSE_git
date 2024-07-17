%% Fiber parameters
% enter here all the relevant parameter for the fiber

function data = Amit_Singlemode_1modes_CW808nm()

data.folder_name = 'Amit_Singlemode_1modes_CW808nm';    % folder where the output will be stored

% wavelength and modes
data.lambda0 = 808e-9;                          % center wavelength [m]
data.lrange = 100e-9;                            % wavelength range [m]. If 0 only the center wavelength will be used
data.Nf = 10;                                     % number of frequency points at which the modes will be calculated
data.num_modes = 1;                              % number of modes to calculate

% fiber size and grid
data.radius = 4.4;                    % outer radius of fiber [um]. has to be round number for the script
data.spatial_window = data.radius * 4;                        % full spatial window size [um]
data.Nx = 1024;                                  % number of spatial grid points


% optical fiber features
data.extra_params.ncore_diff = 1.4590-1.4532;           % difference between the index at the center of the core, and the cl
data.extra_params.alpha = 2.00;                  % Shape parameter
data.extra_params.NA = 0.14;                     % Numerical Aperture

% fiber refractive index type
data.profile_function = @build_step;             % function that builds the fiber ; build_GRIN\build_step

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