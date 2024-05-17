function [num_it_tot, A1_right_end] = GMMNLSE_MPA_SMgain(A0, dt, sim, prefactor, SRa_info, SRb_info, SK_info, D_op, haw, hbw, G, saturation_energy)
%GMMNLSE_MPA_SMGAIN Take one step according to the GMMNLSE, with a 
%total-energy saturating gain model
%
% A0 - initial field, (N, num_modes) matrix, in the frequency domain in W^1/2
% dt - time grid point spacing, in ps
%
% sim.f0 - center frequency, in THz
% sim.sw - 1 includes self-steepening, 0 does not
% sim.deltaZ - small step size, in m
% sim.MPA.M - parallel extent, 1 is no parallelization
% sim.MPA.n_tot_max - maximum number of iterations
% sim.MPA.n_tot_min - minimum number of iterations
% sim.MPA.tol - tolerance for convergence at each iteration
% sim.singe_yes - 1 = single, 0 = double
% sim.gpu_yes - 1 = GPU, 0 = CPU
% sim.lmc.model - 0,1,2 = the model of (random or deterministic) linear mode coupling used
% sim.SK_factor - SK = SK_factor * fiber.SR
% sim.scalar - scalar or polarized fields
% sim.Raman_model - which Raman model is used
%
% nonlin_const - n2*w0/c, in W^-1 m
% SRa_info.SRa - SRa tensor, in m^-2
% SRa_info.nonzero_midx1234s - required SRa indices in total
% SRa_info.nonzero_midx34s - required (SRa) indices for partial Raman term (only for CPU computation)
% SRb_info.SRb - SRb tensor, in m^-2
% SRb_info.nonzero_midx1234s - required SRb indices in total
% SRb_info.nonzero_midx34s - required (SRb) indices for partial Raman term (only for CPU computation)
% mode_info.nonzero_midx34s - required (SRa) indices for partial Raman term
% SK_info.SK - SK tensor, in m^2 (unempty if considering polarizaton modes)
% SK_info.nonzero_midx1234s - required SK indices in total (unempty if considering polarizaton modes)
%
% omegas - angular frequencies in 1/ps, in the fft ordering
%
% Without mode coupling or with mode coupling lmc_model 1:
%   D_op - dispersion operator for all modes and all small steps, with size (N, num_modes)
% With mode coupling lmc_model 1,2:
%   D_op - dispersion operator for all modes and all small steps, with size (num_modes, num_modes, N)
%
% haw - isotropic Raman response in the frequency domain
% hbw - anisotropic Raman response in the frequency domain
% G - gain term, a (N, 1) vector with the gain prefactor for each frequency, in m^-1
% saturation_energy - scale energy in nJ for gain saturation
%
% Output:
% num_it_tot - iteration at the end of which convergence was reached
% A1_right_end - (N, num_modes) matrix with the field at the end of the step, for each mode, in the frequency domain

anisotropic_Raman_included = ~sim.scalar & sim.Raman_model==2;

N = size(A0, 1); % number of time/freq points
num_modes = size(A0, 2);

% 1) Set initial values for psi
psi = repmat(A0, 1, 1, sim.MPA.M+1); % M copies of psi(w,z) = A(w,z), in the frequency domain!

for n_it = 1:sim.MPA.n_tot_max
    % 2) Calculate A(w,z) at all z
    
    % To get from psi to A, we need D_op, which depends on A through the
    % gain saturation
    if sim.lmc.model == 2
        if sim.gpu_yes
            if sim.single_yes
                DG_op = complex(zeros(num_modes, num_modes, N, sim.MPA.M+1, 'single', 'gpuArray'));
                D.pos = complex(ones(num_modes, num_modes, N, sim.MPA.M+1, 'single', 'gpuArray'));
            else
                DG_op = complex(zeros(num_modes, num_modes, N, sim.MPA.M+1, 'gpuArray'));
                D.pos = complex(ones(num_modes, num_modes, N, sim.MPA.M+1, 'gpuArray'));
            end
        else
            if sim.single_yes
                DG_op = complex(zeros(num_modes, num_modes, N, sim.MPA.M+1, 'single'));
                D.pos = complex(ones(num_modes, num_modes, N, sim.MPA.M+1, 'single'));
            else
                DG_op = complex(zeros(num_modes, num_modes, N, sim.MPA.M+1));
                D.pos = complex(ones(num_modes, num_modes, N, sim.MPA.M+1));
            end
        end
    else
        if sim.gpu_yes
            if sim.single_yes
                DG_op = complex(zeros(N, num_modes, sim.MPA.M+1, 'single', 'gpuArray'));
                D.pos = complex(ones(N, num_modes, sim.MPA.M+1, 'single', 'gpuArray'));
            else
                DG_op = complex(zeros(N, num_modes, sim.MPA.M+1, 'gpuArray'));
                D.pos = complex(ones(N, num_modes, sim.MPA.M+1, 'gpuArray'));
            end
        else
            if sim.single_yes
                DG_op = complex(zeros(N, num_modes, sim.MPA.M+1, 'single'));
                D.pos = complex(ones(N, num_modes, sim.MPA.M+1, 'single'));
            else
                DG_op = complex(zeros(N, num_modes, sim.MPA.M+1));
                D.pos = complex(ones(N, num_modes, sim.MPA.M+1));
            end
        end
    end
    if sim.gpu_yes
        if sim.single_yes
            A_w = complex(zeros(N, sim.MPA.M+1, num_modes, 'single', 'gpuArray'));
        else
            A_w = complex(zeros(N, sim.MPA.M+1, num_modes, 'gpuArray'));
        end
    else
        if sim.single_yes
            A_w = complex(zeros(N, sim.MPA.M+1, num_modes, 'single'));
        else
            A_w = complex(zeros(N, sim.MPA.M+1, num_modes));
        end
    end
    
    A_w(:,1,:) = psi(:,:,1); % the first point is unchanged
    if sim.lmc.model == 2
        psi_for_pagefun_or_mmx = permute(psi,[2 4 1 3]); % (num_modes,1,N,M+1)
    end
    cumulative_G_scale = 0; % Keep track of the total gain, determined by the energy at each small step
    for ii = 2:sim.MPA.M+1
        field_ii = permute(A_w(:,ii-1,:),[1 3 2]);
        last_energy = calc_total_energy(field_ii,N,dt); % Apply gain based on the energy at the last small step (in nJ)
        cumulative_G_scale = cumulative_G_scale + 1/(1+last_energy/saturation_energy);
        if sim.lmc.model == 2
            DG_op(:,:,:,ii) = diag_plus((ii-1)*D_op,cumulative_G_scale*G); % Save the dispersion operator to be used later
            if sim.gpu_yes
                D.pos(:,:,:,ii) = myexpm_(DG_op(:,:,:,ii),[],[],true,false,sim.gpu_yes); % For multiple matrices, "myexpm_" is faster than MATLAB internal "expm"
                A_w_tmp = pagefun(@mtimes, D.pos(:,:,:,ii),psi_for_pagefun_or_mmx(:,:,:,ii)); % Apply the linear dispersion term and nonlinear gain term, D.pos
                A_w(:,ii,:) = permute(A_w_tmp,[3 4 1 2]);
            else % CPU
                D.pos(:,:,:,ii) = myexpm_(DG_op(:,:,:,ii),[],[],true,false,sim.gpu_yes); % For multiple matrices, "myexpm_" is faster than MATLAB internal "expm"
                A_w_tmp = mmx_mult(D.pos(:,:,:,ii),psi_for_pagefun_or_mmx(:,:,:,ii)); % Apply the linear dispersion term and nonlinear gain term, D.pos
                A_w(:,ii,:) = permute(A_w_tmp,[3 4 1 2]);
            end
        else
            DG_op(:,:,ii) = (ii-1)*D_op + cumulative_G_scale*G; % Save the dispersion operator to be used later
            D.pos(:,:,ii) = exp(DG_op(:,:,ii));
            A_w(:,ii,:) = D.pos(:,:,ii).*psi(:,:,ii); % Apply the linear dispersion term and nonlinear gain term
        end
    end
    if sim.lmc.model == 2
        D.neg = myexpm_(-DG_op,[],[],true,false,sim.gpu_yes);
    else
        D.neg = exp(-DG_op);
    end

    % 3) Calculate A(t,z) at all z
    A_t = fft(A_w);

    % 4) Calculate Kerr, SRa, and SRb terms
    if sim.gpu_yes
        if sim.single_yes
            Kerr = complex(zeros(N, sim.MPA.M+1, num_modes, 'single', 'gpuArray'));
            Ra = complex(zeros(N, sim.MPA.M+1, num_modes, num_modes, 'single', 'gpuArray'));
        else
            Kerr = complex(zeros(N, sim.MPA.M+1, num_modes, 'gpuArray'));
            Ra = complex(zeros(N, sim.MPA.M+1, num_modes, num_modes, 'gpuArray'));
        end
    else
        if sim.single_yes
            Kerr = complex(zeros(N, sim.MPA.M+1, num_modes, 'single'));
            Ra = complex(zeros(N, sim.MPA.M+1, num_modes, num_modes, 'single'));
        else
            Kerr = complex(zeros(N, sim.MPA.M+1, num_modes));
            Ra = complex(zeros(N, sim.MPA.M+1, num_modes, num_modes));
        end
    end
    if anisotropic_Raman_included
        if sim.gpu_yes
            if sim.single_yes
                Rb = complex(zeros(N, sim.MPA.M+1, num_modes, num_modes, 'single', 'gpuArray'));
            else
                Rb = complex(zeros(N, sim.MPA.M+1, num_modes, num_modes, 'gpuArray'));
            end
        else
            if sim.single_yes
                Rb = complex(zeros(N, sim.MPA.M+1, num_modes, num_modes, 'single'));
            else
                Rb = complex(zeros(N, sim.MPA.M+1, num_modes, num_modes));
            end
        end
    end
    
    if sim.gpu_yes
        % If using the GPU, do the computation with fast CUDA code
        if isempty(SK_info) % scalar fields
            if sim.Raman_model==0
                Kerr = feval(sim.kernel, Kerr, complex(A_t), SRa_info.SRa, SRa_info.nonzero_midx1234s,  N, sim.MPA.M+1, sim.SK_factor, size(SRa_info.nonzero_midx1234s, 2), num_modes);
            else
                [Kerr, Ra] = feval(sim.kernel, Kerr, Ra, complex(A_t), SRa_info.SRa, SRa_info.nonzero_midx1234s,  N, sim.MPA.M+1, sim.SK_factor, size(SRa_info.nonzero_midx1234s, 2), num_modes);
            end
        else % polarized fields
            switch sim.Raman_model
                case 0
                    Kerr = feval(sim.kernel, Kerr, complex(A_t), SK_info.SK, SK_info.nonzero_midx1234s, N, sim.MPA.M+1, size(SK_info.nonzero_midx1234s, 2), num_modes);
                case 1
                    [Kerr, Ra] = feval(sim.kernel, Kerr, Ra, complex(A_t), SRa_info.SRa, SRa_info.nonzero_midx1234s, SK_info.SK, SK_info.nonzero_midx1234s, N, sim.MPA.M+1, size(SRa_info.nonzero_midx1234s, 2), size(SK_info.nonzero_midx1234s, 2), num_modes);
                case 2
                    [Kerr, Ra, Rb] = feval(sim.kernel, Kerr, Ra, Rb, complex(A_t), SRa_info.SRa, SRa_info.nonzero_midx1234s, SRb_info.SRb, SRb_info.nonzero_midx1234s, SK_info.SK, SK_info.nonzero_midx1234s, N, sim.MPA.M+1, size(SRa_info.nonzero_midx1234s, 2), size(SK_info.nonzero_midx1234s, 2), num_modes);
            end
        end
    else
        % ---------------------------------------------------------------------
        % Original lines:
        %
        % Since it's easier to understand the code with this for-loop, I
        % keep it here; whereas the vectorized code implemented below is
        % faster.
        %
        %    If using the CPU, first precompute Ra_mn
        %    for nz_idx = 1:size(SRa_info.nonzero_midx34s, 2)
        %        midx3 = SRa_info.nonzero_midx34s(1, nz_idx);
        %        midx4 = SRa_info.nonzero_midx34s(2, nz_idx);
        %        Ra_mn(:, :, midx3, midx4) = A_t(:, :, midx3).*conj(A_t(:, :, midx4));
        %    end
        %    
        %    Then calculate the num_modes^4 sum
        %    for nz_idx = 1:size(SRa_info.nonzero_midx1234s, 2)
        %        midx1 = SRa_info.nonzero_midx1234s(1, nz_idx);
        %        midx2 = SRa_info.nonzero_midx1234s(2, nz_idx);
        %        midx3 = SRa_info.nonzero_midx1234s(3, nz_idx);
        %        midx4 = SRa_info.nonzero_midx1234s(4, nz_idx);
        %
        %        Kerr(:, :, midx1) = Kerr(:, :, midx1) + sim.SK_factor*SRa_info.SR(nz_idx)*A_t(:, :, midx2).*A_t(:, :, midx3).*conj(A_t(:, :, midx4));
        %        Ra(:, :, midx1, midx2) = Ra(:, :, midx1, midx2) + SRa_info.SRa(nz_idx)*Ra_mn(:, :, midx3, midx4);
        %    end
        % -----------------------------------------------------------------
        % Vectorize the above code below.
        % 
        % If using the CPU, first precompute Ra_mn and Rb_mn.
        if sim.Raman_model~=0
            if n_it == 1
                midx34s_sub2ind = @(x)...
                    cellfun(@(xx)...
                        feval(@(sub) sub2ind(num_modes*ones(1,2),sub{:}), num2cell(xx)),... % this extra "feval" is to get "xx", which is of the size 2x1, into the input arguments of "sub2ind", so transforming "xx" into a 2x1 cell, each containing an integer, and using {:} expansion is necessary
                    mat2cell(x,2,ones(1,size(x,2)))); % transform (2,num_nonzero34) midx34s into linear indices of a num_modes-by-num_modes matrix
                    % What "midx34s_sub2ind" does (e.g.):
                    %
                    %   x = [1 3;
                    %        5 4]
                    %
                    %   After "mat2cell": {[1;  {[3;  (2x1 cells, each having 2x1 array)
                    %                       5]}   4]}
                    %
                    %   First,
                    %
                    %   xx = {[1;  , then after "num2cell": {{1}; (1 cell with 2x1 cell)
                    %          5]}                           {5}}
                    %
                    %   The purpose of separating 1 and 5 into cells is to use
                    %   index expansion, {:}, to put them into the input
                    %   arguments of "sub2ind" function.
                    %
                    %   For 6 modes and thus for 6x6 matrix, sub2ind([6 6],1,5) = 25
                    %
                    %   Do the same for xx = {[3;  and get sub2ind([6 6],3,4) = 21
                    %                          4]}
                    %   Finally, midx34s_sub2ind = [25 21] (1x2 array)

                SRa_nonzero_midx34s = midx34s_sub2ind(SRa_info.nonzero_midx34s); % the corresponding linear indices of the 3rd-dimensional "num_nonzero34" above
                if anisotropic_Raman_included
                    SRb_nonzero_midx34s = midx34s_sub2ind(SRb_info.nonzero_midx34s); % the corresponding linear indices of the 3rd-dimensional "num_nonzero34" above
                end
            end
            Ra_mn = A_t(:, :, SRa_info.nonzero_midx34s(1,:)).*conj(A_t(:, :, SRa_info.nonzero_midx34s(2,:))); % (N,M+1,num_nonzero34)
            if anisotropic_Raman_included
                Rb_mn = A_t(:, :, SRb_info.nonzero_midx34s(1,:)).*conj(A_t(:, :, SRb_info.nonzero_midx34s(2,:))); % (N,M+1,num_nonzero34)
            end
        end
        
        % Then calculate Kerr,Ra,Rb.
        for midx1 = 1:num_modes
            if isempty(SK_info)
                nz_midx1 = find( SRa_info.nonzero_midx1234s(1,:)==midx1 );
                midx2 = SRa_info.nonzero_midx1234s(2,nz_midx1);
                midx3 = SRa_info.nonzero_midx1234s(3,nz_midx1);
                midx4 = SRa_info.nonzero_midx1234s(4,nz_midx1);
                Kerr(:,:,midx1) = sum(sim.SK_factor*permute(SRa_info.SRa(nz_midx1),[3 2 1]).*A_t(:, :, midx2).*A_t(:, :, midx3).*conj(A_t(:, :, midx4)),3);
            else
                nz_midx1 = find( SK_info.nonzero_midx1234s(1,:)==midx1 );
                midx2 = SK_info.nonzero_midx1234s(2,nz_midx1);
                midx3 = SK_info.nonzero_midx1234s(3,nz_midx1);
                midx4 = SK_info.nonzero_midx1234s(4,nz_midx1);
                Kerr(:,:,midx1) = sum(permute(SK_info.SK(nz_midx1),[3 2 1]).*A_t(:, :, midx2).*A_t(:, :, midx3).*conj(A_t(:, :, midx4)),3);
            end
            if sim.Raman_model ~= 0
                % Ra
                for midx2 = 1:num_modes
                    if ~isempty(SK_info)
                        nz_midx1 = find( SRa_info.nonzero_midx1234s(1,:)==midx1 );
                    end
                    nz_midx = nz_midx1( SRa_info.nonzero_midx1234s(2,nz_midx1)==midx2 ); % all the [midx1;midx2;?;?]
                    midx3 = SRa_info.nonzero_midx1234s(3,nz_midx);
                    midx4 = SRa_info.nonzero_midx1234s(4,nz_midx);
                    idx = midx34s_sub2ind([midx3;midx4]); % the linear indices
                    idx = arrayfun(@(i) find(SRa_nonzero_midx34s==i,1), idx); % the indices connecting to the 3rd-dimensional "num_nonzero34" of Ra_mn
                    Ra(:, :, midx1, midx2) = sum(permute(SRa_info.SRa(nz_midx),[3 2 1]).*Ra_mn(:, :, idx),3);
                end
                % Rb
                if anisotropic_Raman_included
                    for midx2 = 1:num_modes
                        nz_midx1 = find( SRb_info.nonzero_midx1234s(1,:)==midx1 );
                        nz_midx = nz_midx1( SRb_info.nonzero_midx1234s(2,nz_midx1)==midx2 ); % all the [midx1;midx2;?;?]
                        midx3 = SRb_info.nonzero_midx1234s(3,nz_midx);
                        midx4 = SRb_info.nonzero_midx1234s(4,nz_midx);
                        idx = midx34s_sub2ind([midx3;midx4]); % the linear indices
                        idx = arrayfun(@(i) find(SRb_nonzero_midx34s==i,1), idx); % the indices connecting to the 3rd-dimensional "num_nonzero34" of Rb_mn
                        Rb(:, :, midx1, midx2) = sum(permute(SRb_info.SRb(nz_midx),[3 2 1]).*Rb_mn(:, :, idx),3);
                    end
                end
            end
        end
        if anisotropic_Raman_included
            clear Ra_mn Rb_mn
        elseif sim.Raman_model ~= 0
            clear Ra_mn
        end
    end
    
    % 5,6) Apply the convolution for each part of the Raman sum
    % The convolution using Fourier Transform is faster if both arrays are
    % large. If one of the array is small, "conv" can be faster.
    % Please refer to
    % "https://blogs.mathworks.com/steve/2009/11/03/the-conv-function-and-implementation-tradeoffs/"
    % for more information.
    if sim.Raman_model ~= 0
        Ra = dt*fft(haw.*ifft(Ra));

        if ~anisotropic_Raman_included
            % 7) Finish the sum for the Raman term, and add everything together
            %    nonlinear(:, :, midx1) = nonlinear(:, :, midx1) + Ra(:, :, midx1, midx2).*A_t(:, :, midx2);
            nonlinear = Kerr + sum(Ra.*permute(A_t,[1 2 4 3]),4);
        else % polarized fields with an anisotropic Raman
            Rb = dt*fft(hbw.*ifft(Rb));

            nonlinear = Kerr + sum((Ra+Rb).*permute(A_t,[1 2 4 3]),4);
        end
    else
        nonlinear = Kerr;
    end
    
    % 8) Take the fourier transform for each z, p
    % 9) Sum for each z, and save the intermediate results for the next iteration
    
    % Multiply the nonlinear factor
    nonlinear = prefactor.*permute(ifft(nonlinear), [1 3 2]); % (N, num_modes, M+1)
    
    % Incorporate deltaZ and D_neg term for trapezoidal integration.
    if sim.lmc.model == 2
        if sim.gpu_yes
            % Although "expm" is time consuming, "mldivide" will encounter
            % "singular matrix" problem; therefore, I compute D.neg first
            % above and do a matrix multiplication.
            % There's no "singualr matrix" problem with "nogain" propagation,
            % I think the addition of gain term somehow messes up "D.pos".
            nonlinear = permute(nonlinear,[2 4 1 3]); % (num_modes,1,N,M+1)
            %nonlinear = sim.deltaZ*pagefun(@mldivide, D.pos, nonlinear);
            nonlinear = sim.deltaZ*pagefun(@mtimes, D.neg,nonlinear);
            nonlinear = permute(nonlinear,[3 1 4 2]); % (N, num_modes, M+1)
        else % CPU
            nonlinear = permute(nonlinear,[2 4 1 3]); % (num_modes,1,N,M+1)
            nonlinear = sim.deltaZ*mmx_mult(D.neg,nonlinear);
            nonlinear = permute(nonlinear,[3 1 4 2]); % (N, num_modes, M+1)
        end
    else
        nonlinear = sim.deltaZ*D.neg.*nonlinear; % (N, num_modes, M+1)
    end

    % Save the previous psi at the right end, then compute the new psi's
    last_psi = psi(:, :, sim.MPA.M+1);
    % "psi" is calculated from trapezoidal integrals, which results in
    % (1/2,1,1,......,1,1/2) coefficients.
    % Note that for each z plane, it has a (1/2, 1, 1...,1/2) factor in nonlinear.
    nonlinear(:,:,1) = nonlinear(:,:,1)/2;
    psi(:,:,2:end) = psi(:,:,1) + cumsum(nonlinear(:,:,1:sim.MPA.M),3) + nonlinear(:,:,2:end)/2;
    
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
        error('Error in GMMNLSE_MPA_SMgain: The step did not converge after %d iterations, aborting.', sim.MPA.n_tot_max);
    end
    
    % 10) Psi has been updated at all z_j, so now repeat n_tot times
end

% 11) Get back to A from psi at the right edge
if sim.lmc.model == 2
    if sim.gpu_yes
        current_psi = permute(current_psi,[2 3 1]); % (num_modes,1,N)
        A1_right_end = pagefun(@mtimes, D.pos(:,:,:,sim.MPA.M+1),current_psi);
        A1_right_end = permute(A1_right_end,[3 1 2]);
    else % CPU
        current_psi = permute(current_psi,[2 3 1]); % (num_modes,1,N)
        A1_right_end = mmx_mult(D.pos(:,:,:,sim.MPA.M+1),current_psi);
        A1_right_end = permute(A1_right_end,[3 1 2]);
    end
else
    A1_right_end = D.pos(:,:,sim.MPA.M+1).*current_psi;
end

end

function A = diag_plus(A,B)
%DIAG_PLUS It does addition only on the diagonal terms of A to an array of
%B.

n = size(A,1);
m = size(A,3); % the number of matrices

diag_idx = (0:n^2:(n^2*(m-1)))' + (1:(n+1):n^2); % find out the indices of the diagonal elements
A(diag_idx) = A(diag_idx) + B;

end

function total_energy = calc_total_energy(A,N,dt)
%total_energy = sum(trapz(abs(fftshift(A,1)).^2))*N*dt/1e3; % in nJ

center_idx = ceil(size(A,1)/2);
A2 = abs(A).^2;

total_energy = (sum(sum(A2))-sum(sum(A2([center_idx,center_idx+1],:))))*N*dt/1e3; % in nJ;

end