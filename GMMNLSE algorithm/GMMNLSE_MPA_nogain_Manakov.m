function [num_it_tot, A1_right_end] = GMMNLSE_MPA_nogain_Manakov(A0, dt, sim, prefactor, kappaK,kappaR1,kappaR2, D, hrw, mode_groups)
%GMMNLSE_MPA_NOGAIN_MANAKOV Take one step according to the Manakov 
%GMMNLSE, without gain
%
% A0 - initial field, (N, num_modes) matrix, in the frequency domain in W^1/2
% dt - time grid point spacing, in ps
%
% sim.f0 - center frequency, in THz
% sim.sw - 1 includes self-steepening, 0 does not
% sim.deltaZ - small step size, in m
% sim.MPA.M - parallel extent, 1 is no parallelization
% sim.n_tot_max - maximum number of iterations
% sim.n_tot_min - minimum number of iterations
% sim.tol - tolerance for convergence at each iteration
% sim.singe_yes - 1 = single, 0 = double
% sim.gpu_yes - 1 = GPU, 0 = CPU
%
% nonlin_const - n2*w0/c, in W^-1 m
% kappaK - the nonlinear coefficient for Kerr term
% kappaR1 - one of the nonlinear coefficient for Raman term
% kappaR2 - another nonlinear coefficient for Raman term
%
% omegas - angular frequencies in 1/ps, in the fft ordering
%
% D.pos - exp(Dz) for all modes and all small steps, with size (N, num_modes, M+1)
% D.neg - exp(-Dz) for all modes and all small steps, with size (N, num_modes, M+1)
%
% hrw - Raman response in the frequency domain
% mode_groups - the index of modes in each mode group
%
% Output:
% num_it_tot - iteration at the end of which convergence was reached
% A1_right_end - (N, num_modes) matrix with the field at the end of the step, for each mode, in the frequency domain

N = size(A0, 1); % number of time/freq points
num_modes = size(A0, 2);
num_mode_groups = length(sim.lmc.mode_groups);

% 1) Set initial values for psi
psi = repmat(A0, 1, 1, sim.MPA.M+1); % M copies of psi(w,z) = A(w,z), in the frequency domain

% Iterate to solve for psi
for n_it = 1:sim.MPA.n_tot_max
    % 2) Calculate A(w,z) at all z
    A_w = D.pos.*psi;
    
    % 3) Calculate A(t,z) at all z
    A_t = fft(A_w);

    % 4) Calculate AK, AR1, AR2 terms, the field summation terms of Kerr
    % and Raman effects
    if sim.gpu_yes
        if sim.single_yes
            Aaa = complex(zeros(N,num_mode_groups,sim.MPA.M+1,'single','gpuArray'));
            AR2 = complex(zeros(N,num_modes,sim.MPA.M+1,num_mode_groups,'single','gpuArray'));
        else
            Aaa = complex(zeros(N,num_mode_groups,sim.MPA.M+1,'gpuArray'));
            AR2 = complex(zeros(N,num_modes,sim.MPA.M+1,num_mode_groups,'gpuArray'));
        end
    else
        if sim.single_yes
            Aaa = complex(zeros(N,num_mode_groups,sim.MPA.M+1,'single'));
            AR2 = complex(zeros(N,num_modes,sim.MPA.M+1,num_mode_groups,'single'));
        else
            Aaa = complex(zeros(N,num_mode_groups,sim.MPA.M+1));
            AR2 = complex(zeros(N,num_modes,sim.MPA.M+1,num_mode_groups));
        end
    end
    % Compute sum(|Aa|^2,a) term
    for mg = 1:num_mode_groups
        Aaa(:,mg,:) = sum(abs(A_t(:,mode_groups{mg},:)).^2, 2);
    end
    Aaa = permute(Aaa,[1 4 3 2]); % (N,1,sim.MPA.M+1,num_mode_group)
    
    % Kerr term
    AK = A_t.*Aaa; % (N,num_modes,sim.MPA.M+1,num_mode_group)

    % Raman term:
    % The convolution using Fourier Transform is faster if both arrays are
    % large. If one of the array is small, "conv" can be faster.
    % Please refer to
    % "https://blogs.mathworks.com/steve/2009/11/03/the-conv-function-and-implementation-tradeoffs/"
    % for more information.
    % AR1
    AR1aa = dt*fft(hrw.*ifft(Aaa));
    AR1 = A_t.*AR1aa; % (N,num_modes,sim.MPA.M+1,num_mode_group)
    % AR2
    AR2qa = dt*fft(hrw.*ifft(A_t.*permute(conj(A_t),[1 4 3 2])));
    A_for_aqa_summation = permute(A_t,[1 4 3 2]); % (N,1,sim.MPA.M+1,num_mode_group)
    for mg = 1:num_mode_groups
        AR2(:,:,:,mg) = sum(A_for_aqa_summation(:,:,:,mode_groups{mg}).*AR2qa(:,:,:,mode_groups{mg}) ,4); % (N,num_modes,sim.MPA.M+1,num_mode_group)
    end
    
    % 5) Sum up all the AK,AR1,AR2 terms for nonlinear effects
    % The summation between kappa and A? is processed on the 4th index, the
    % mode-group one.
    UpK = sum(kappaK.*AK,4); % Kerr
    
    UpR1 = sum(kappaR1.*AR1,4);
    UpR2 = sum(kappaR2.*AR2,4);
    UpR = UpR1+UpR2; % Raman
    
    Up = UpK + UpR; % all nonlinear terms
    
    % 6) Take the Fourier transform for each z, p
    Up = ifft(Up);

    % 7) Sum for each z, and save the intermediate results for the next iteration
    
    % Multiply the nonlinear factor
    Up = prefactor.*Up;
    
    % Incorporate deltaZ and D.neg term for trapezoidal integration.
    Up = sim.deltaZ*D.neg.*Up;
    
    % Save the previous psi at the right end, then compute the new psi's
    last_psi = psi(:, :, sim.MPA.M+1);
    % "psi" is calculated from trapezoidal integrals, which results in
    % (1/2,1,1,......,1,1/2) coefficients.
    % Note that for each z plane, it has a (1/2, 1, 1...,1/2) factor in Up.
    Up(:,:,1) = Up(:,:,1)/2;
    psi(:,:,2:end) = psi(:,:,1) + cumsum(Up(:,:,1:sim.MPA.M),3) + Up(:,:,2:end)/2;

    % Calculate the average NRMSE = take the RMSE between the previous psi
    % and the current psi at the right edge, normalize by the absolute max,
    % and average over all modes
    current_psi = psi(:,:,sim.MPA.M+1);
    abs_current_psi = abs(current_psi);
    energy_current_psi = sum(abs_current_psi.^2);
    weight = energy_current_psi/sum(energy_current_psi);
    NRMSE_p = sqrt(sum(abs(current_psi-last_psi).^2)/N) ./ max(abs_current_psi).*weight;
    NRMSE_p(isnan(NRMSE_p)) = 0; % exclude modes with all zero fields
    avg_NRMSE = sum(NRMSE_p);
    
    if sim.verbose
        fprintf('iteration %d, avg NRMSE: %f\n', n_it, avg_NRMSE)
    end
    
    % If it has converged to within tol, then quit
    if avg_NRMSE < sim.MPA.tol && n_it >= sim.MPA.n_tot_min
        num_it_tot = n_it; % Save the number of iterations it took
        break
    end
    
    if n_it == sim.MPA.n_tot_max
        error('Error in GMMNLSE_MPA_step: The step did not converge after %d iterations, aborting.', sim.MPA.n_tot_max);
    end
    
    % 8) Psi has been updated at all z_j, so now repeat n_tot times
end

% 9) Get back to A from psi at the right edge
A1_right_end = D.pos(:,:,sim.MPA.M+1).*current_psi;

end