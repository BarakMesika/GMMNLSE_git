function [fiber, sim, input_field, others] = GIF300_10modes_532nm(E_tot,dz)


        %% Other Prameters
        others.data_folder = 'GIF300_10ns_E_varification\'; % where to save the propagation data


        
        %% Fiber parameters
        fiber.MM_folder = '../Fibers/GIF300_10modes_534nm/';               % Fiber data folder
        fiber.L0 = 20;                                                                 % Fiber length [m]
        fiber.n2 =2.3e-20;                                                              % non linear coeff [m^2/W]
        % fiber.gain_Aeff = ;                                                           % deffault is 1.6178e-10
        
        %% Simulation parameters
        time_window = 100e3;                                                          % Time Window [ps]
        N = 2^14;                                                                  % the number of time points
        save_num = 100;                                                                % how many popagation points to save


        sim.deltaZ = dz;                                                            % delta z point [m] 
        sim.single_yes = false;                                                          % for GPU. use true
        sim.adaptive_deltaZ.model = 0;                                                  % turn adaptive-step off
        sim.step_method = "RK4IP";                                                      % use "MPA" instead of the default "RK4IP"
        % sim.MPA.M = 10;                                                                   % if we use MPA algorithem
        sim.Raman_model = 0;                                                            % Raman 
        sim.gpu_yes = true;                                                             % enable GPU optimization
        sim.gain_model = 0;                                                             % gain modle. 0 to disable
        sim.progress_bar = true;                                                        % disable for slightly better preformence
        sim.sw = 0;                                                                     % disable shock waves (self stipening)

    
        %% DONT EDIT
        Fiber_params = load([fiber.MM_folder 'Fiber_params.mat'], 'data');
        fiber.radius = Fiber_params.data.radius;
        others.modes = Fiber_params.data.num_modes; 
        others.Nx = Fiber_params.data.Nx;
        dt = time_window/N;
        t = (-N/2:N/2-1)'*dt; % time axsis [ps]
        input_field.dt = dt;
        input_field.fields = zeros([size(t,1) others.modes]);
        E_modes = zeros(1,others.modes);

        others.numeric.time_window = time_window;
        others.numeric.N = N;
        others.numeric.deltaZ = sim.deltaZ;
         %% Initial Pulse 
        % noise to the intial pulse
        % noise = randn(size(t))+1i*randn(size(t));
        % noise = noise/sqrt( dt*sum(abs(noise).^2)*1e-3 )*sqrt(1e-6);
        noise = 0;

		T0 = 10e3 / ( 2*sqrt(log(2)) );               % 175fs FWHM ( 1/2 NO 1/e)
        tmp = exp(-(1/2)*(t/T0).^2);                    % init pulse shape (will be notmalized to 1nJ)
         
        % input_field.E_tot = 1e9;                                      % Total Energy [pJ]
        input_field.E_tot = E_tot;
        E_modes(1) = 0.2; E_modes(2) = 0.2; E_modes(3) = 0.2;
        E_modes(4) = 0.2; E_modes(5) = 0.2;
        

        %%  DONT EDIT 
        % sets the other parameters
        fiber.betas_filename = 'betas.mat';                                        % betas file name (in the fiber data folder)
        fiber.S_tensors_filename = ['S_tensors_' num2str(others.modes) 'modes.mat'];      % S Tensor file (in the fiber data folder)


        sim.cuda_dir_path = '../cuda';                          % add cuda folder
        sim.save_period = fiber.L0/save_num;
        sim.lambda0 = Fiber_params.data.lambda0;                          % central wavelength [m]

        [fiber,sim] = load_default_GMMNLSE_propagate(fiber,sim,'multimode');


        f = ifftshift( (-N/2:N/2-1)'/N/dt + sim.f0 ); % in the order of "omegas" in the "GMMNLSE_propagate.m"
        c = 299792.458; % [nm/ps];
        others.lambda = c./f; % [nm]
        others.t = t;
        others.f = f;

        % normlized energy to 1pJ
        tmp = tmp/sqrt( dt*sum(abs(tmp).^2));

        E_modes = E_modes * input_field.E_tot;

        % give energy to the initial pulse. for each mode
        for ii=1:others.modes
            input_field.fields(:,ii) = sqrt(E_modes(ii))*tmp + noise;
        end


        % dispersion and non linear lengths caculations
        w0 = 2*pi*sim.f0;
        nonlin_const = fiber.n2*w0/2.99792458e-4; % W^-1 m
        gammaLP01 = nonlin_const*fiber.SR(1,1,1,1);
        % P0 = abs(fiber.betas(3,1))/gammaLP01/(T0.^2);
        P0 = max( abs( input_field.fields(:,1) ).^2 );

        Ld = T0^2/abs(fiber.betas(3,1));
        Lnl = (gammaLP01 * P0)^(-1);

end