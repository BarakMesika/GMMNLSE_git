function [fiber, sim, input_field, others] = Barak_GIF625_3modes_CW1550nm_1550nm()
        %% Other Prameters
        others.data_folder = 'BarakTest_multimode_soliton\'; % where to save the propagation data
        
        %% Fiber parameters
        fiber.MM_folder = '../Fibers/Barak_GIF625_3center_modes_CW1550nm_1550nm/';               % Fiber data folder
        fiber.L0 = 50;                                                                 % Fiber length [m]
        fiber.n2 = 3.2e-20;                                                              % non linear coeff [m^2/W]
        % fiber.gain_Aeff = ;                                                           % deffault is 1.6178e-10
        
        %% Simulation parameters
        time_window = 50;                                                          % Time Window [ps]
        N = 2^14;                                                                  % the number of time points
        save_num = 250;                                                                % how many popagation points to save


        sim.deltaZ = 500e-6;                                                            % delta z point [m] 
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

         %% Initial Pulse 

		T0 = 0.3; % 300fs
        tmp = sech(t/T0);                    % init pulse shape (will be notmalized to 1nJ)
        % tmp =  exp(-(t/T0).^2);

        input_field.E_tot = 500;                                      % Total Energy [pJ]
        E_modes(1) = 1/3; E_modes(2) = 1/3; E_modes(3) = 1/3; 
        
        

        




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

        % normlized energy to 1pJ
        tmp = tmp/sqrt( dt*sum(abs(tmp).^2) );

        E_modes = E_modes * input_field.E_tot;

        % give energy to the initial pulse. for each mode
        for ii=1:others.modes
            input_field.fields(:,ii) = sqrt(E_modes(ii))*tmp;
        end


end