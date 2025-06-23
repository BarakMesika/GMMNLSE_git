clc; clear;  close all;


E_tot = [1e7];
L0 = 10;
N=2^13;
dz = 10 * 1e-6;

for i=1:length(E_tot)
        tic;
        [fiber, sim, input_field, others] = GIF625_520nm_NanoSecPulse(L0,N,dz,E_tot(i));
        PropagationScript_fun( fiber, sim, input_field, others )
        fprintf("time for iter  %d is -  %.4f seconds\n", (i) ,toc)
end

