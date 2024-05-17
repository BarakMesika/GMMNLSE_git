function [num_it_tot, A1_right_end] = GMMNLSE_MPA_newgain(A0, dt, sim, prefactor, SRa_info, SRb_info, SK_info, D, haw, hbw, G, saturation_intensity, fr)
%GMMNLSE_MPA_NEWGAIN Take one step according to the GMMNLSE, with a 
%spatially saturating gain model
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
% SK_info.SK - SK tensor, in m^2 (unempty if considering polarizaton modes)
% SK_info.nonzero_midx1234s - required SK indices in total (unempty if considering polarizaton modes)
%
% omegas - angular frequencies in 1/ps, in the fft ordering
%
% Without mode coupling or with mode coupling lmc.model 1:
%   D.pos - exp(Dz) for all modes and all small steps, with size (N, num_modes, M+1)
%   D.neg - exp(-Dz) for all modes and all small steps, with size (N, num_modes, M+1)
% With mode coupling lmc.model 2,3:
%   D.pos - exp(Dz) for all modes and all small steps, with size (num_modes, num_modes, N, M+1)
%
% haw - isotropic Raman response in the frequency domain
% hbw - anisotropic Raman response in the frequency domain
% G - gain term, a (N, 1) vector with the gain prefactor for each frequency, in m^-1
% saturation_intensity - scale intensity in J m^-2 for gain saturation
%
% Output:
% num_it_tot - iteration at the end of which convergence was reached
% A1_right_end - (N, num_modes) matrix with the field at the end of the step, for each mode, in the frequency domain

anisotropic_Raman_included = ~sim.scalar & sim.Raman_model==2;

N = size(A0, 1); % number of time/freq points
num_modes = size(A0, 2);

% Explanation about the computation of the transfer matrix when considering
% polarization modes:
%   transfer_matrix and Bmn are in (M+1,num_spatial_modes,num_spatial_modes,2)
%   "2" represents two polarization modes.
%   I put the odd modes(e.g., x-polarization) in the 1st one, the even(e.g., y-polarization) in the 2nd one.
%   Now there are two sub-arrays, each related to different polarizations,
%   then I can do the multiplication for each polarization and matrix division later.
if sim.scalar && sim.lmc.model==0
    num_spatial_modes = num_modes;
    polar = 1;
else
    num_spatial_modes = num_modes/2;
    polar = 2;
end

% 1) Set initial values for psi
psi = repmat(A0, 1, 1, sim.MPA.M+1); % M copies of psi(w,z) = A(w,z), in the frequency domain!

for n_it = 1:sim.MPA.n_tot_max
    % 2) Calculate A(w,z) at all z:
    %    A_w = D.pos*psi
    if sim.lmc.model==2 % use linear mode coupling: lmc.model 2
        if sim.gpu_yes % GPU
            psi_for_pagefun = permute(psi,[2 4 1 3]); % (num_modes,1,N,M+1)
            A_w = pagefun(@mtimes, D.pos,psi_for_pagefun);
            A_w = permute(A_w, [3 4 1 2]); % (N, M+1, num_modes)
        else % CPU
            psi_for_mmx_mult = permute(psi,[2 4 1 3]); % (num_modes,1,N,M+1)
            A_w = mmx_mult(D.pos,psi_for_mmx_mult);
            A_w = permute(A_w, [3 4 1 2]); % (N, M+1, num_modes)
        end
    else
        A_w = permute(D.pos.*psi, [1 3 2]); % (N, M+1, num_modes)
    end

    % 3) Calculate A(t,z) at all z
    A_t = fft(A_w);

    % 4) Calculate U_p(t,z) = SK term, and V_pl(t,z) = SR term
    % If not using the GPU, we will precompute Ra_mn and Rb_mn before the num_modes^4 sum
    
    % Setup the matrices, extra are needed to store the transfer matrix
    if sim.gpu_yes
        if sim.single_yes
            Kerr = complex(zeros(N, sim.MPA.M+1, num_modes, 'single', 'gpuArray'));
            Ra = complex(zeros(N, sim.MPA.M+1, num_modes, num_modes, 'single', 'gpuArray'));
            
            transfer_matrix = complex(zeros(sim.MPA.M+1, num_spatial_modes, num_spatial_modes, polar, 'single', 'gpuArray'));
        else
            Kerr = complex(zeros(N, sim.MPA.M+1, num_modes, 'gpuArray'));
            Ra = complex(zeros(N, sim.MPA.M+1, num_modes, num_modes, 'gpuArray'));
            
            transfer_matrix = complex(zeros(sim.MPA.M+1, num_spatial_modes, num_spatial_modes, polar, 'gpuArray'));
        end
    else
        if sim.single_yes
            Kerr = complex(zeros(N, sim.MPA.M+1, num_modes, 'single'));
            Ra = complex(zeros(N, sim.MPA.M+1, num_modes, num_modes, 'single'));
            
            transfer_matrix = complex(zeros(sim.MPA.M+1, num_spatial_modes, num_spatial_modes, polar, 'single'));
        else
            Kerr = complex(zeros(N, sim.MPA.M+1, num_modes));
            Ra = complex(zeros(N, sim.MPA.M+1, num_modes, num_modes));
            
            transfer_matrix = complex(zeros(sim.MPA.M+1, num_spatial_modes, num_spatial_modes, polar));
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

    % Calculate Bmn for all l and m, and M
    % This is faster to do in MATLAB and only scales like num_modes^2
    %    Bmn(:, midx3, midx4,p) = squeeze(dt/1e12*sum(A_t(:, :, midx3,p).*conj(A_t(:, :, midx4,p)), 1))/saturation_intensity;
    if sim.scalar
        Bmn = dt/1e12*squeeze(sum(A_t.*conj(permute(A_t,[1 2 4 3]))))/saturation_intensity;
    else
        A_t_for_Bmn = cat(5,A_t(:,:,1:2:num_modes-1),A_t(:,:,2:2:num_modes)); % separate the polarization modes
        Bmn = shiftdim(dt/1e12*sum(A_t_for_Bmn.*conj(permute(A_t_for_Bmn,[1 2 4 3 5])))/saturation_intensity, 1);
    end
    if sim.Raman_model ~= 0
        Bmn = Bmn/fr; % This is to fix that SRa has been multiplied by fiber.fr in GMMNLSE_propagate.m in advance, which we need only SRa value for transfer_matrix
    end
    
    % Calculate the tensors
    if sim.gpu_yes
        % If using the GPU, do the computation with fast CUDA code
        % The gain transfer matrix can be calculated at the same time
        if isempty(SK_info) % scalar fields
            if sim.Raman_model==0
                [Kerr, transfer_matrix] = feval(sim.kernel, Kerr, transfer_matrix, complex(A_t), complex(Bmn), SRa_info.SRa, SRa_info.nonzero_midx1234s,  N, sim.MPA.M+1, sim.SK_factor, size(SRa_info.nonzero_midx1234s, 2), num_modes);
            else
                [Kerr, Ra, transfer_matrix] = feval(sim.kernel, Kerr, Ra, transfer_matrix, complex(A_t), complex(Bmn), SRa_info.SRa, SRa_info.nonzero_midx1234s,  N, sim.MPA.M+1, sim.SK_factor, size(SRa_info.nonzero_midx1234s, 2), num_modes);
            end
        else % polarized fields
            switch sim.Raman_model
                case 0
                    [Kerr, transfer_matrix] = feval(sim.kernel, Kerr, transfer_matrix, complex(A_t), complex(Bmn), SRa_info.SRa, SRa_info.nonzero_midx1234s, SK_info.SK, SK_info.nonzero_midx1234s, N, sim.MPA.M+1, size(SRa_info.nonzero_midx1234s, 2), size(SK_info.nonzero_midx1234s, 2), num_modes);
                case 1
                    [Kerr, Ra, transfer_matrix] = feval(sim.kernel, Kerr, Ra, transfer_matrix, complex(A_t), complex(Bmn), SRa_info.SRa, SRa_info.nonzero_midx1234s, SK_info.SK, SK_info.nonzero_midx1234s, N, sim.MPA.M+1, size(SRa_info.nonzero_midx1234s, 2), size(SK_info.nonzero_midx1234s, 2), num_modes);
                case 2
                    [Kerr, Ra, Rb, transfer_matrix] = feval(sim.kernel, Kerr, Ra, Rb, transfer_matrix, complex(A_t), complex(Bmn), SRa_info.SRa, SRa_info.nonzero_midx1234s, SRb_info.SRb, SRb_info.nonzero_midx1234s, SK_info.SK, SK_info.nonzero_midx1234s, N, sim.MPA.M+1, size(SRa_info.nonzero_midx1234s, 2), size(SK_info.nonzero_midx1234s, 2), num_modes);
            end
        end
    else
        % -----------------------------------------------------------------
        % Original lines(for only scalar fields):
        %
        % Since it's easier to understand the code with this for-loop, I
        % keep it here; whereas the vectorized code implemented below is
        % faster.
        %
        %    If using the CPU, first precompute Ra_mn
        %    for nz_idx = 1:size(SRa_info.nonzero_midx34s, 2)
        %       midx3 = SRa_info.nonzero_midx34s(1, nz_idx);
        %       midx4 = SRa_info.nonzero_midx34s(2, nz_idx);
        %       Ra_mn(:, :, midx3, midx4) = A_t(:, :, midx3).*conj(A_t(:, :, midx4));
        %    end
        %
        %    Then calculate the num_modes^4 sum
        %    for nz_idx = 1:size(SRa_info.nonzero_midx1234s, 2)
        %        midx1 = SRa_info.nonzero_midx1234s(1, nz_idx);
        %        midx2 = SRa_info.nonzero_midx1234s(2, nz_idx);
        %        midx3 = SRa_info.nonzero_midx1234s(3, nz_idx);
        %        midx4 = SRa_info.nonzero_midx1234s(4, nz_idx);
        %
        %        Kerr(:, :, midx1) = Kerr(:, :, midx1) + sim.SK_factor*SRa_info.SRa(nz_idx)*A_t(:, :, midx2).*A_t(:, :, midx3).*conj(A_t(:, :, midx4));
        %        Ra(:, :, midx1, midx2) = Ra(:, :, midx1, midx2) + SRa_info.SRa(nz_idx)*Ra_mn(:, :, midx3, midx4);
        %        transfer_matrix(:, midx1, midx4) = transfer_matrix(:, midx1, midx4) + SRa_info.SRa(nz_idx)*Blm(:, midx2, midx3);
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
            % Kerr
            if isempty(SK_info)
                nz_midx1 = find( SRa_info.nonzero_midx1234s(1,:)==midx1 );
                midx2 = SRa_info.nonzero_midx1234s(2,nz_midx1);
                midx3 = SRa_info.nonzero_midx1234s(3,nz_midx1);
                midx4 = SRa_info.nonzero_midx1234s(4,nz_midx1);
                Kerr(:,:,midx1) = sum(permute(sim.SK_factor*SRa_info.SRa(nz_midx1),[3 2 1]).*A_t(:, :, midx2).*A_t(:, :, midx3).*conj(A_t(:, :, midx4)),3);
            else
                nz_midx1 = find( SK_info.nonzero_midx1234s(1,:)==midx1 );
                midx2 = SK_info.nonzero_midx1234s(2,nz_midx1);
                midx3 = SK_info.nonzero_midx1234s(3,nz_midx1);
                midx4 = SK_info.nonzero_midx1234s(4,nz_midx1);
                Kerr(:,:,midx1) = sum(permute(SK_info.SK(nz_midx1),[3 2 1]).*A_t(:, :, midx2).*A_t(:, :, midx3).*conj(A_t(:, :, midx4)),3);
            end
            if sim.Raman_model~=0
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
        elseif sim.Raman_model~=0
            clear Ra_mn
        end
        
        % Transfer matrix:
        % Explanation for the computation below:
        %   transfer_matrix: T, Bmn: B
        %   S_plmn: (p,l) relates to the indices of T
        %           (m,n) relates to the indices of B
        %   I calculate first the linear indices of (p,l) and (m,n), use
        %   these two indices to do the multiplication: T=S*B
        if sim.scalar
            lidx_T = sub2ind(num_spatial_modes*ones(1,2),SRa_info.nonzero_midx1234s(1,:)',SRa_info.nonzero_midx1234s(2,:)');
            lidx_B = sub2ind(num_spatial_modes*ones(1,2),SRa_info.nonzero_midx1234s(3,:)',SRa_info.nonzero_midx1234s(4,:)');
            msize_T = num_spatial_modes^2;
        else
            % odd, even represents polarizations, which gives an extra
            % addition of "num_spatial_modes^2" for the linear indices of
            % the even modes.
            odd_or_even = @(x) rem(x+1,2);
            % Recovery back to spatial mode indices to calculate the
            % linear indices related to T and B.
            recover_spatial_midx = @(x) ceil(x/2);
            
            oddeven_T = double(odd_or_even(SRa_info.nonzero_midx1234s(1,:)'));
            oddeven_B = double(odd_or_even(SRa_info.nonzero_midx1234s(3,:)'));
            lidx = recover_spatial_midx(SRa_info.nonzero_midx1234s);
            
            lidx_T = sub2ind(num_spatial_modes*ones(1,2),lidx(1,:)',lidx(2,:)') + oddeven_T*num_spatial_modes^2;
            lidx_B = sub2ind(num_spatial_modes*ones(1,2),lidx(3,:)',lidx(4,:)') + oddeven_B*num_spatial_modes^2;
            
            msize_T = num_spatial_modes^2*2;
        end
        for midx = 1:msize_T
            idx_T = lidx_T==midx;
            idx_B = lidx_B(idx_T);
            transfer_matrix(:,midx) = sum(SRa_info.SRa(idx_T)'.*squeeze(Bmn(:,idx_B)),2);
        end
    end
    
    % Finish the transfer_matrix by adding the diagonals and reshaping
    transfer_matrix = permute(transfer_matrix, [2, 3, 1, 4]) + eye(num_spatial_modes); % now it goes (num_spatial_modes, num_spatial_modes, M+1)
    A_w = permute(A_w, [1, 3, 2]); % A_w goes (N, num_modes, M+1)

    % Invert and apply the transfer function for each mode
    gain_rhs = permute(G.*A_w,[2 1 3]); % (num_modes,N,sim.MPA.M+1) matrix, each column is a single w value
    if ~sim.scalar
        gain_rhs = cat(4,gain_rhs(1:2:num_modes-1,:,:),gain_rhs(2:2:num_modes,:,:));
        % "recovery_idx" is used to put the separated polarization modes
        % back into the same dimension of array.
        recovery_idx = [(1:num_spatial_modes);(1:num_spatial_modes)+num_spatial_modes];
        recovery_idx = recovery_idx(:);
    end
    if sim.gpu_yes
        gain_term = permute(pagefun(@mldivide, transfer_matrix,gain_rhs),[2 1 3 4]);
    else % CPU
        gain_term = permute(multbslash(transfer_matrix,gain_rhs),[2 1 3 4]);
    end
    % Put the separated polarization modes back into the same dimension of array.
    if ~sim.scalar
        gain_term = cat(2,gain_term(:,:,:,1),gain_term(:,:,:,2));
        gain_term = gain_term(:,recovery_idx,:);
    end
    
    % 5,6) Apply the convolution for the isotropic Raman
    % The convolution using Fourier Transform is faster if both arrays are
    % large. If one of the array is small, "conv" can be faster.
    % Please refer to
    % "https://blogs.mathworks.com/steve/2009/11/03/the-conv-function-and-implementation-tradeoffs/"
    % for more information.
    if sim.Raman_model~=0
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
    
    % Calculate the full integrand in frequency space, with both nonlinearities
    % Multiply the nonlinear factor
    nonlinear = prefactor.*permute(ifft(nonlinear),[1 3 2]); % (N,num_modes,M+1)
    
    % Incorporate deltaZ and D.neg term for trapezoidal integration.
    if sim.lmc.model == 2
        if sim.gpu_yes
            nonlinear = permute(nonlinear,[2 4 1 3]); % (num_modes,1,N,M+1)
            gain_term = permute(gain_term,[2 4 1 3]); % (num_modes,1,N,M+1)
            nonlinear = pagefun(@mldivide, D.pos, sim.deltaZ*(nonlinear + gain_term));
            nonlinear = permute(nonlinear,[3 1 4 2]); % (N, num_modes, M+1)
        else % CPU
            nonlinear = permute(nonlinear,[2 4 1 3]); % (num_modes,1,N,M+1)
            gain_term = permute(gain_term,[2 4 1 3]); % (num_modes,1,N,M+1)
            nonlinear = multbslash(D.pos, sim.deltaZ*(nonlinear + gain_term));
            nonlinear = permute(nonlinear,[3 1 4 2]); % (N, num_modes, M+1)
        end
    else
        nonlinear = D.neg.*(sim.deltaZ*(nonlinear+gain_term)); % (N, num_modes, M+1)
    end

    % Save the previous psi at the right end, then compute the new psis
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
        error('Error in GMMNLSE_MPA_step: The step did not converge after %d iterations, aborting.', sim.MPA.n_tot_max);
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
