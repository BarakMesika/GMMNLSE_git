% Note that the final pump power and the inversion, N2, are the same at
% each point of the gain fiber for co- and counter-pumping because the gain
% can't respond to the pulse in time and sees only the average effect
% including both the forward and backward propagating pulses.
close all; clearvars;

output_field = [];

filename = 'linear_oscillator.mat';

load(filename);

addpath('../../');

ASE_out = struct('spectrum',ASE_forward_out,'t_rep',gain_param{1}{1}.t_rep);
analyze_field(t,f,output_field(:,1,end),pi/6,1e-6,true,ASE_out);

func = analyze_sim;
func.analyze_fields_within_cavity(t,f,field{end},saved_z,splice_z);
func.analyze_gain_within_cavity(saved_z,splice_z,pump{end},N2{end});

ASE_plot.forward = ASE.forward{end}; ASE_plot.backward = ASE.backward{end};
func.analyze_ASE_within_cavity(f,ASE_plot,saved_z,splice_z);