function mode_time_profiles_tm = decompose_into_modes(normalized_mode_space_profiles_xym, full_field_txy, dx)
%DECOMPOSE_INTO_MODES Get the modal decomposition from the full 3D field
% mode_space_profile - a (Nx, Nx, num_modes) matrix with the mode profile 
% for each mode.
%
% =========================================================================
% Input:
% normalized_mode_space_profiles_xym - a (Nx, Nx, num_modes) matrix with each mode's profile in space. The units do not matter.
% dx - spatial grid spacing, in m
% full_field - a (Nt,Nx,Nx) matrix with full spatiotemporal fields in each time.
% -------------------------------------------------------------------------
% Output:
% mode_time_profiles - a (Nt, num_modes) matrix with each mode's time profile.
% =========================================================================

normalized_mode_space_profiles_mxy = permute(normalized_mode_space_profiles_xym,[3 1 2]);

num_modes = size(normalized_mode_space_profiles_mxy, 1);
Nx = size(normalized_mode_space_profiles_mxy, 2);
Nt = size(full_field_txy, 1);

% Einstein summation convention: tm=(txy)*(mxy)
mode_time_profiles_tm = reshape(full_field_txy,[Nt,Nx^2])*reshape(normalized_mode_space_profiles_mxy,[num_modes,Nx^2]).'*dx^2;

end