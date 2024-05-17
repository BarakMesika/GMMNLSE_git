function full_field_txy = recompose_into_space(normalized_mode_space_profiles_xym, mode_time_profiles_tm)
% RECOMPOSE_INTO_SPACE Combine a set of mode time profiles with their 
% corresponding space profiles to get the full 3D spatio-temperal field.
%
% =========================================================================
% Input:
% normalized_mode_space_profiles_xym - a (Nx, Nx, num_modes) matrix with each mode's profile in space. The units do not matter.
% mode_time_profiles_tm - a (Nt, num_modes) matrix with each mode's time profile.
% -------------------------------------------------------------------------
% Output:
% full_field - a (Nt,Nx,Nx) matrix with full spatiotemporal fields in each time.
% =========================================================================

normalized_mode_space_profiles_mxy = permute(normalized_mode_space_profiles_xym,[3 1 2]);

num_modes = size(normalized_mode_space_profiles_mxy, 1);
Nx = size(normalized_mode_space_profiles_mxy, 2);
Nt = size(mode_time_profiles_tm, 1);

% I use tensor product here, summing over index m.
% Since MATLAB doesn't support tensor product, we need to use the trick
% of "reshape" function with a combination of matrix multiplications.
%
% F_txy = sum( F_tm*P_mxy ,m)
full_field_txy = reshape(...
    mode_time_profiles_tm*reshape(normalized_mode_space_profiles_mxy, [num_modes Nx^2]),...
                    [Nt Nx Nx]);

end