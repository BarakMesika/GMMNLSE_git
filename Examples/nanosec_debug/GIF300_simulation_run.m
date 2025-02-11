clc; clear;

E_tot = [5e7, 1e8, 5e8, 1e9] .* 5.5; 
L0 = 5;
N=2^12;
dZ = 10 * 1e-6;

for i=1:length(E_tot)
    for j=1:length(dZ)
        tic;
        [fiber, sim, input_field, others] = GIF3000_10modes_532nm(L0,N,dz,E_tot(i));
        PropagationScript_fun( fiber, sim, input_field, others )
        fprintf("time for iter  %d is -  %.4f seconds\n", (i+j) ,toc)
    end
end

