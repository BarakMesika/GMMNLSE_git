function [epsilon, x, dx] = build_Nufern(lambda, Nx, spatial_window, radius, extra_params)
% build_Nufern  Build a Nufern-type gain fiber refractive index profile 
% lambda - the wavelength, in um
% Nx - the number of spatial points in each dimension
% spatial_window - the total size of space in each dimension, in um
% radius - the outer radius of the fiber, in um
%
% extra_params.ncore1_diff - the amount to add to the Sellmeier result to get n after the first step
% extra_params.ncore2_diff - the amount to add to the ncore1 to get n after the second step (in the center)
% extra_params.alpha_middle - the shape parameter for the inner slope
% extra_params.alpha_outer - the shape parameter for the outer slope
% extra_params.r_inner - the radius for which the index is flat
% extra_params.r_middle - the radius at which the index becomes parabolic


% Using the Sellmeier equation to generate n(lambda)
a1=0.6961663;
a2=0.4079426;
a3=0.8974794;
b1= 0.0684043;
b2=0.1162414;
b3=9.896161;

nsi=(1+a1*(lambda.^2)./(lambda.^2 - b1^2)+a2*(lambda.^2)./(lambda.^2 - b2^2)+a3*(lambda.^2)./(lambda.^2 - b3^2)).^(0.5);

% There are two steps in this fiber
ncl = nsi; % cladding index
nco_outer = nsi + extra_params.ncore1_diff; % core1 index
nco_inner = nco_outer + extra_params.ncore2_diff; % core2 index


dx = spatial_window/Nx; % um

x = (-Nx/2:Nx/2-1)*dx;
[X, Y] = meshgrid(x, x);
P = sqrt(X.^2 + Y.^2);

% Start with the flat top
epsilon = ones(Nx, Nx)*nco_inner^2;

% Then add the first graded section
middle = find(P > extra_params.r_inner);
epsilon(middle) = (nco_inner - (extra_params.ncore2_diff)*((P(middle)-extra_params.r_inner)/(extra_params.r_middle - extra_params.r_inner)).^extra_params.alpha_middle).^2;

% Then the second graded section
middle2 = find(P > extra_params.r_middle);
epsilon(middle2) = (nco_outer - (extra_params.ncore1_diff)*(P(middle2)/radius).^extra_params.alpha_outer).^2;

% Then the cladding
outer = P > radius;
epsilon(outer) = ncl^2;


end