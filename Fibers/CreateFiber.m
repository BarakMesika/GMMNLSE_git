%% create new fiber
clc; clear;close;
addpath('CreateFiberMatlabFiles/');
%% Get fiber parameters
% change it acording to the requested fiber
data = Amit_Singlemode_1modes_CW808nm();

%% 1. Define the fiber and solve for the modes
solve_for_modes_fun(data);
% keep_wanted_modes(data,[1,4,11])

%% 2. Calculate the dispersion coefficients
calc_dispersion_fun(data);

%% 3. Calculte the overlap tenors
calc_SRSK_tensors_fun(data);

%% Enjoy your new Fiber :)



