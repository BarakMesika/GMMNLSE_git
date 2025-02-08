clc; clear;
% origial_pulse = 220;
% small_pulse = 10;
% multi_factor = small_pulse / origial_pulse;
% 
% E_tot = [50] * 1e9; % [pJ]
% E_tot = E_tot .* multi_factor;

E_tot = [0.5, 1, 5, 10]*1e9 / 20;

dZ = [10] * 1e-6;

for i=1:length(E_tot)
    for j=1:length(dZ)
        tic;
        [fiber, sim, input_field, others] = GIF3000_10modes_532nm(E_tot(i),dZ(j));
        PropagationScript_fun( fiber, sim, input_field, others )
        fprintf("time for iter  %d is -  %.4f seconds\n", (i+j) ,toc)
    end
end

