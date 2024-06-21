clc; clear;

%% Need to run once
[fiber, sim, input_field, others] = TL_KBSC_15modes_1030nm_testing(1e3,zeros(1,15));


modes = others.modes; 

gain_rate_eqn.mode_volume = modes; 

% Gain info - TODO
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


% Gain parameters - TODO
[gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN] = ...
                                                gain_info( sim,gain_rate_eqn,others.lambda );
% send this into GMMNLSE_propagate function
gain_param = {gain_rate_eqn,cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN}; 


%%
E_vec = load('modes_coeff.mat');
E_vec = E_vec.modes_coeff;
E_tot_vec = linspace(10e3,30e3,10);


for iter = 1:length(E_tot_vec)

% pulse energy
E_tot = E_tot_vec(iter); 


% run propagation

[fiber, sim, input_field, others] = TL_KBSC_15modes_1030nm_testing(E_tot,E_vec);

dirName  = others.data_folder;                        % new folder to save the data
mkdir(dirName);


% start propagation
output_field = GMMNLSE_propagate(fiber,input_field,sim,gain_param);

fileCount = iter;
fName = ['data_' num2str(fileCount,'%03.0f')];          % data file name
save([dirName fName], 'output_field', 'fiber', 'sim', 'input_field', 'others');


%% plots

dt = input_field.dt;
t = others.t;
lambda = others.lambda;
cmap = linspecer(others.modes);

E = squeeze( sum(abs(output_field.fields).^2,1) )*dt;
distance = 0:sim.save_period:fiber.L0; 

group = zeros(5,length(E(1,:)));
group(1,:) = E(1,:);
group(2,:) = ( E(2,:) + E(3,:) ) / 2;
group(3,:) = ( E(4,:) + E(5,:) + E(6,:) ) / 3;
group(4,:) = ( E(7,:) + E(8,:) + E(9,:) + E(10,:) ) / 4; 
group(5,:) = ( E(11,:) + E(12,:) + E(13,:) + E(14,:) + E(15,:) ) / 5; 

figure;
for ii=1:size(group,1)
    plot(distance, group(ii,:),'DisplayName', ['group:' num2str(ii)], 'LineWidth', 2,...
        'Color', cmap(ii,:));
    hold on
end
hold off
legend
xlabel('Propagation length (m)');
ylabel('Energy (pJ)');
title({['file no.' num2str(iter)], ['Energy = ' num2str(E_tot * 1e-3) 'nJ'] });
ylim([0 input_field.E_tot])
grid on;

saveas(gcf ,[dirName fName], 'jpg')
close;

end