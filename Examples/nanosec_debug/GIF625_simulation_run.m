clc; clear;  close all;


E_tot = [0.05 0.1 0.5 1 2 5 ]*1e9 ;
L0 = 15;
N=2^12;
dz = 10 * 1e-6;

for i=1:length(E_tot)
    for j=1:length(dz)
        tic;
        [fiber, sim, input_field, others] = GIF625_520nm_NanoSecPulse(L0,N,dz,E_tot(i));
        PropagationScript_fun( fiber, sim, input_field, others )
        fprintf("time for iter  %d is -  %.4f seconds\n", (i+j) ,toc)
    end
end

