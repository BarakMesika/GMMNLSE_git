clc; clear;

E_tot = [ 5e6,5e7, 1e8] .* 5.5; 
L0 = 5;
N=2^14;
dZ = 10 * 1e-6;
propagation_print = false;

for i=1:length(E_tot)
    tic;
    fprintf("iter %d started \n", i);
    [fiber, sim, input_field, others] = GIF300_10modes_532nm(L0,N,dZ,E_tot(i));
    PropagationScript_fun( fiber, sim, input_field, others, propagation_print)
    fprintf("time for iter  %d is -  %.4f seconds\n", (i) ,toc)
end

