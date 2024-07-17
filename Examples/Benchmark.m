clc; clear;

%%
fiber_script = @TL_KBSC_10modes_1030nm;
energy_vec = linspace(10e3, 50e3, 5);

% run from the sub folder. the section add to the path the main simulation folder
%% Add the folders of multimode files and others
addpath('../');                                         % add where many GMMNLSE-related functions like  "GMMNLSE_propagate" is

energy_input = energy_vec;

%%%%%%%%
% START TIMER
%%%%%%%%
tic;
timer_iter = zeros(1, length(energy_input));

for ii=1:length(energy_input)
[fiber, sim, input_field, others] = fiber_script(energy_input(ii));
disp(['START ITERATION '  num2str(ii)]);

modes = others.modes; 

gain_rate_eqn.mode_volume = modes;                  % the total number of available spatial modes in the fiber



%% Gain info - TODO
gain_rate_eqn.MM_folder = fiber.MM_folder; % specify the folder with the eigenmode profiles
gain_rate_eqn.cross_section_filename = 'Liekki Yb_AV_20160530.txt';
gain_rate_eqn.saved_mat_filename = 'MM_YDFA_strong_waveguide_rate_eqn'; % the files used to temporarily store the data during iterations if necessary
gain_rate_eqn.reuse_data = false; % For a ring or linear cavity, the pulse will enter a steady state eventually.
                                  % If reusing the pump and ASE data from the previous roundtrip, the convergence can be much faster, especially for counterpumping.
gain_rate_eqn.linear_oscillator_model = 0; % For a linear oscillator, there are pulses from both directions simultaneously, which will deplete the gain;
                                           % therefore , the backward-propagating pulses need to be taken into account.
gain_rate_eqn.core_diameter = fiber.radius*2; % um
gain_rate_eqn.cladding_diafffmeter = 125; % um
gain_rate_eqn.core_NA = 0.02; % in fact, this is only used in single-mode
gain_rate_eqn.absorption_wavelength_to_get_N_total = 915; % nm
gain_rate_eqn.absorption_to_get_N_total = 1.7; % dB/m
gain_rate_eqn.pump_wavelength = 976; % nm
gain_rate_eqn.copump_power = 0; % W
gain_rate_eqn.counterpump_power = 0; % W
gain_rate_eqn.midx = 1:modes; % the mode index
gain_rate_eqn.mode_volume = modes; % the total number of available spatial modes in the fiber
gain_rate_eqn.downsampling_factor = 1; % downsample the eigenmode profiles to run faster
gain_rate_eqn.t_rep = 1/15e6; % assume 15MHz here; s; the time required to finish a roundtrip (the inverse repetition rate of the pulse)
                             % This gain model solves the gain of the fiber under the steady-state condition; therefore, the repetition rate must be high compared to the lifetime of the doped ions.
gain_rate_eqn.tau = 840e-6; % lifetime of Yb in F_(5/2) state (Paschotta et al., "Lifetme quenching in Yb-doped fibers"); in "s"
gain_rate_eqn.export_N2 = true; % whether to export N2, the ion density in the upper state or not
gain_rate_eqn.ignore_ASE = true;
gain_rate_eqn.max_iterations = 10; % For counterpumping or considering ASE, iterations are required.
gain_rate_eqn.tol = 1e-5; % the tolerance for the iteration
gain_rate_eqn.allow_coupling_to_zero_fields = false; % set this to false all the time for now
gain_rate_eqn.verbose = true; % show the information(final pulse energy) during iterations of computing the gain

% TODO - check that
gain_rate_eqn.cladding_diameter = gain_rate_eqn.cladding_diafffmeter;


%% Gain parameters - TODO
[gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN] = ...
                                                gain_info( sim,gain_rate_eqn,others.lambda );
% send this into GMMNLSE_propagate function
gain_param = {gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN}; 


%% Propagation

dirName  = others.data_folder;                        % new folder to save the data
mkdir(dirName);


% start propagation
output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);

% save data
% uamp = single( output_field.fields );

% make sure we create a new saved file
fileCount = 0;
folder_files = dir(dirName);
for i = 1:length(folder_files)
    % Check if the entry is not a directory
    if ~folder_files(i).isdir
        fileCount = fileCount + 1;
    end
end

fName = ['data_' num2str(fileCount+1,'%03.0f')];          % data file name
save([dirName fName], 'output_field', 'fiber', 'sim', 'input_field', 'others');

toc
timer_iter(ii) = toc;
end
