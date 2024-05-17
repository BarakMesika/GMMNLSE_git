function [A1,a5,...
          opt_deltaZ,success] = GMMNLSE_RK4IP_taylorgain_adaptive(A0, dt, sim, prefactor, SRa_info, SRb_info, SK_info, D_op, haw, hbw, a5_1, G, saturation_intensity, fr)
%GMMNLSE_RK4IP_TAYLORGAIN_ADAPTIVE Take one step according to the GMMNLSE, with a spatially saturating gain model based on a Taylor expansion
% A0 - initial field, (N, num_modes) matrix, in the frequency domain in W^1/2
% dt - time grid point spacing, in ps
%
% sim.f0 - center frequency, in THz
% sim.sw - 1 includes self-steepening, 0 does not
% sim.deltaZ - step size, in m
% sim.singe_yes - 1 = single, 0 = double
% sim.gpu_yes - 1 = GPU, 0 = CPU
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
% D - dispersion term for all modes in m^-1, with size (N, num_modes)
% haw - isotropic Raman response in the frequency domain
% hbw - anisotropic Raman response in the frequency domain
% G - gain term, a (N, 1) vector with the gain prefactor for each frequency, in m^-1
% saturation_intensity - scale intensity in J m^-2 for gain saturation
%
% Output:
% A1 - (N, num_modes) matrix with the field after the step, for each mode, in the frequency domain

N = size(A0, 1); % number of time/freq points
num_modes = size(A0, 2);

% Explanation about the computation of the transfer matrix when considering
% polarization modes:
%   transfer_matrix and Bmn are in (M+1,num_spatial_modes,num_spatial_modes,2)
%   "2" represents two polarization modes.
%   I put the odd modes(e.g., x-polarization) in the 1st one, the even(e.g., y-polarization) in the 2nd one.
%   Now there are two sub-arrays, each related to different polarizations,
%   then I can do the multiplication for each polarization and matrix division later.
if sim.scalar
    num_spatial_modes = num_modes;
    polar = 1;
else
    num_spatial_modes = num_modes/2;
    polar = 2;
end

anisotropic_Raman_included = ~sim.scalar & sim.Raman_model==2;

% Setup the matrices
if sim.gpu_yes
    if sim.single_yes
        Kerr = complex(zeros(N, num_modes, 'single', 'gpuArray'));
        Ra = complex(zeros(N, num_modes, num_modes, 'single', 'gpuArray'));
        
        transfer_matrix = complex(zeros(num_spatial_modes, num_spatial_modes, polar, 'single', 'gpuArray'));
    else
        Kerr = complex(zeros(N, num_modes, 'gpuArray'));
        Ra = complex(zeros(N, num_modes, num_modes, 'gpuArray'));
        
        transfer_matrix = complex(zeros(num_spatial_modes, num_spatial_modes, polar, 'gpuArray'));
    end
else
    if sim.single_yes
        Kerr = complex(zeros(N, num_modes, 'single'));
        Ra = complex(zeros(N, num_modes, num_modes, 'single'));
        
        transfer_matrix = complex(zeros(num_spatial_modes, num_spatial_modes, polar, 'single'));
    else
        Kerr = complex(zeros(N, num_modes));
        Ra = complex(zeros(N, num_modes, num_modes));
        
        transfer_matrix = complex(zeros(num_spatial_modes, num_spatial_modes, polar));
    end
end
if anisotropic_Raman_included
    if sim.gpu_yes
        if sim.single_yes
            Rb = complex(zeros(N, num_modes, num_modes, 'single', 'gpuArray'));
        else
            Rb = complex(zeros(N, num_modes, num_modes, 'gpuArray'));
        end
    else
        if sim.single_yes
            Rb = complex(zeros(N, num_modes, num_modes, 'single'));
        else
            Rb = complex(zeros(N, num_modes, num_modes));
        end
    end
else
    Rb = [];
end

if ~sim.gpu_yes
    if sim.single_yes
        dA_dz_gain = complex(zeros(N, num_spatial_modes, polar, 'single'));
    else
        dA_dz_gain = complex(zeros(N, num_spatial_modes, polar));
    end
else
    dA_dz_gain = [];
end

% used for gain term
if ~sim.scalar
    % "recovery_idx" is used to put the separated polarization modes
    % back into the same dimension of array.
    recovery_idx = [(1:num_spatial_modes);(1:num_spatial_modes)+num_spatial_modes];
    recovery_idx = recovery_idx(:);
else
    recovery_idx = [];
end

D = D_op*sim.deltaZ/2;
expD = exp(D);

% 1) Represented under the interaction picture
A_IP = expD.*A0;

% 2) Propagate through the nonlinearity
if isempty(a5_1)
    a5_1 = N_op(       A0,                     dt, sim, SK_info, SRa_info, SRb_info, Kerr, Ra, Rb, haw, hbw, anisotropic_Raman_included, prefactor, polar, saturation_intensity, transfer_matrix, fr, recovery_idx, G, dA_dz_gain, N, num_modes, num_spatial_modes);
end
a1 = expD.*a5_1;
a2 =       N_op(       A_IP+a1*(sim.deltaZ/2), dt, sim, SK_info, SRa_info, SRb_info, Kerr, Ra, Rb, haw, hbw, anisotropic_Raman_included, prefactor, polar, saturation_intensity, transfer_matrix, fr, recovery_idx, G, dA_dz_gain, N, num_modes, num_spatial_modes);
a3 =       N_op(       A_IP+a2*(sim.deltaZ/2), dt, sim, SK_info, SRa_info, SRb_info, Kerr, Ra, Rb, haw, hbw, anisotropic_Raman_included, prefactor, polar, saturation_intensity, transfer_matrix, fr, recovery_idx, G, dA_dz_gain, N, num_modes, num_spatial_modes);
a4 =       N_op(expD.*(A_IP+a3*(sim.deltaZ)),  dt, sim, SK_info, SRa_info, SRb_info, Kerr, Ra, Rb, haw, hbw, anisotropic_Raman_included, prefactor, polar, saturation_intensity, transfer_matrix, fr, recovery_idx, G, dA_dz_gain, N, num_modes, num_spatial_modes);

A1 = expD.*(A_IP + (a1+2*a2+2*a3)*(sim.deltaZ/6)) + a4*(sim.deltaZ/6);

% 3) Local error estimate
a5 =       N_op(       A1,                     dt, sim, SK_info, SRa_info, SRb_info, Kerr, Ra, Rb, haw, hbw, anisotropic_Raman_included, prefactor, polar, saturation_intensity, transfer_matrix, fr, recovery_idx, G, dA_dz_gain, N, num_modes, num_spatial_modes);
err = sum(abs((a4-a5)*(sim.deltaZ/10)).^2,1);

% 4) Stepsize control
normA = sum(abs(A1).^2,1);
err = max(sqrt(err./normA));
opt_deltaZ = max(0.5,min(2,0.8*(sim.adaptive_deltaZ.threshold/err)^(1/4)))*sim.deltaZ;

success = err < sim.adaptive_deltaZ.threshold;

end

function dAdz = N_op(A_w, dt, sim,...
                     SK_info, SRa_info, SRb_info,...
                     Kerr, Ra, Rb,...
                     haw, hbw, anisotropic_Raman_included,...
                     prefactor, polar,...
                     saturation_intensity, transfer_matrix, fr, recovery_idx, G, dA_dz_gain,...
                     N, num_modes, num_spatial_modes)
%N_op Calculate dAdz

A_t = fft(A_w);

% Calculate Bmn for all l and m
% This is faster to do in MATLAB and only scales like num_modes^2
%    Bmn(midx3, midx4,p) = squeeze(dt/1e12*sum(A_t(:, midx3,p).*conj(A_t(:, midx4,p)), 1))/saturation_intensity;
if sim.scalar
    Bmn = dt/1e12*squeeze(sum(A_t.*conj(permute(A_t,[1 3 2]))))/saturation_intensity;
else
    A_t_for_Bmn = cat(4,A_t(:,1:2:num_modes-1),A_t(:,2:2:num_modes)); % separate the polarization modes
    Bmn = shiftdim(dt/1e12*sum(A_t_for_Bmn.*conj(permute(A_t_for_Bmn,[1 3 2 4])))/saturation_intensity, 1);
end
if sim.Raman_model ~= 0
    Bmn = Bmn/fr; % This is to fix that SRa has been multiplied by fiber.fr in GMMNLSE_propagate.m in advance, which we need only SRa value for transfer_matrix
end

% Calculate the large num_modes^4 sum term
if sim.gpu_yes
    % If using the GPU, do the computation with fast CUDA code
    % The gain transfer matrix can be calculated at the same time
    if isempty(SK_info) % scalar fields
        if sim.Raman_model==0
            [Kerr, transfer_matrix] = feval(sim.kernel, Kerr, transfer_matrix, complex(A_t), complex(Bmn), SRa_info.SRa, SRa_info.nonzero_midx1234s,  N, 1, sim.SK_factor, size(SRa_info.nonzero_midx1234s, 2), num_modes);
        else
            [Kerr, Ra, transfer_matrix] = feval(sim.kernel, Kerr, Ra, transfer_matrix, complex(A_t), complex(Bmn), SRa_info.SRa, SRa_info.nonzero_midx1234s,  N, 1, sim.SK_factor, size(SRa_info.nonzero_midx1234s, 2), num_modes);
        end
    else % polarized fields
        switch sim.Raman_model
            case 0
                [Kerr, transfer_matrix] = feval(sim.kernel, Kerr, transfer_matrix, complex(A_t), complex(Bmn), SRa_info.SRa, SRa_info.nonzero_midx1234s, SK_info.SK, SK_info.nonzero_midx1234s, N, 1, size(SRa_info.nonzero_midx1234s, 2), size(SK_info.nonzero_midx1234s, 2), num_modes);
            case 1
                [Kerr, Ra, transfer_matrix] = feval(sim.kernel, Kerr, Ra, transfer_matrix, complex(A_t), complex(Bmn), SRa_info.SRa, SRa_info.nonzero_midx1234s, SK_info.SK, SK_info.nonzero_midx1234s, N, 1, size(SRa_info.nonzero_midx1234s, 2), size(SK_info.nonzero_midx1234s, 2), num_modes);
            case 2
                [Kerr, Ra, Rb, transfer_matrix] = feval(sim.kernel, Kerr, Ra, Rb, transfer_matrix, complex(A_t), complex(Bmn), SRa_info.SRa, SRa_info.nonzero_midx1234s, SRb_info.SRb, SRb_info.nonzero_midx1234s, SK_info.SK, SK_info.nonzero_midx1234s, N, 1, size(SRa_info.nonzero_midx1234s, 2), size(SK_info.nonzero_midx1234s, 2), num_modes);
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
    %       Ra_mn(:, midx3, midx4) = A_t(:, midx3).*conj(A_t(:, midx4));
    %    end
    %
    %    Then calculate the num_modes^4 sum
    %    for nz_idx = 1:size(SRa_info.nonzero_midx1234s, 2)
    %        midx1 = SRa_info.nonzero_midx1234s(1, nz_idx);
    %        midx2 = SRa_info.nonzero_midx1234s(2, nz_idx);
    %        midx3 = SRa_info.nonzero_midx1234s(3, nz_idx);
    %        midx4 = SRa_info.nonzero_midx1234s(4, nz_idx);
    %
    %        Kerr(:, midx1) = Kerr(:, midx1) + sim.SK_factor*SRa_info.SRa(nz_idx)*A_t(:, midx2).*A_t(:, midx3).*conj(A_t(:, midx4));
    %        Ra(:, midx1, midx2) = Ra(:, midx1, midx2) + SRa_info.SRa(nz_idx)*Ra_mn(:, midx3, midx4);
    %    end
    % -----------------------------------------------------------------
    % Vectorize the above code below.
    % 
    % If using the CPU, first precompute Ra_mn and Rb_mn.
    if sim.Raman_model~=0
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
        Ra_mn = A_t(:, SRa_info.nonzero_midx34s(1,:)).*conj(A_t(:, SRa_info.nonzero_midx34s(2,:))); % (N,num_nonzero34)
        if anisotropic_Raman_included
            SRb_nonzero_midx34s = midx34s_sub2ind(SRb_info.nonzero_midx34s); % the corresponding linear indices of the 3rd-dimensional "num_nonzero34" above
            Rb_mn = A_t(:, SRb_info.nonzero_midx34s(1,:)).*conj(A_t(:, SRb_info.nonzero_midx34s(2,:))); % (N,num_nonzero34)
        end
    end
    
    % Then calculate Kerr,Ra,Rb.
    for midx1 = 1:num_modes
        % Kerr
        if isempty(SK_info) % scalar fields
            nz_midx1 = find( SRa_info.nonzero_midx1234s(1,:)==midx1 );
            midx2 = SRa_info.nonzero_midx1234s(2,nz_midx1);
            midx3 = SRa_info.nonzero_midx1234s(3,nz_midx1);
            midx4 = SRa_info.nonzero_midx1234s(4,nz_midx1);
            Kerr(:,midx1) = sum(permute(sim.SK_factor*SRa_info.SRa(nz_midx1),[2 1]).*A_t(:, midx2).*A_t(:, midx3).*conj(A_t(:, midx4)),2);
        else % polarized fields
            nz_midx1 = find( SK_info.nonzero_midx1234s(1,:)==midx1 );
            midx2 = SK_info.nonzero_midx1234s(2,nz_midx1);
            midx3 = SK_info.nonzero_midx1234s(3,nz_midx1);
            midx4 = SK_info.nonzero_midx1234s(4,nz_midx1);
            Kerr(:,midx1) = sum(permute(SK_info.SK(nz_midx1),[2 1]).*A_t(:, midx2).*A_t(:, midx3).*conj(A_t(:, midx4)),2);
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
                idx = arrayfun(@(i) find(SRa_nonzero_midx34s==i,1), idx); % the indices connecting to the 2nd-dimensional "num_nonzero34" of Ra_mn
                Ra(:, midx1, midx2) = sum(permute(SRa_info.SRa(nz_midx),[2 1]).*Ra_mn(:, idx),2);
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
                    Rb(:, midx1, midx2) = sum(permute(SRb_info.SRb(nz_midx),[2 1]).*Rb_mn(:, idx),2);
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
        transfer_matrix(midx) = SRa_info.SRa(idx_T)'*squeeze(Bmn(idx_B));
    end
end

% Finish the transfer_matrix by adding the diagonals
transfer_matrix = eye(num_spatial_modes) - transfer_matrix; % now it goes (num_spatial_modes, num_spatial_modes)

% Calculate h*Ra as F-1(h F(Ra))
% The convolution using Fourier Transform is faster if both arrays are
% large. If one of the array is small, "conv" can be faster.
% Please refer to
% "https://blogs.mathworks.com/steve/2009/11/03/the-conv-function-and-implementation-tradeoffs/"
% for more information.
if sim.Raman_model~=0
    Ra = dt*fft(haw.*ifft(Ra));

    if ~anisotropic_Raman_included
        % Finish the sum for the Raman term, and add eveything together
        %    Kerr(:, midx1) = Kerr(:, midx1) + Ra(:, midx1, midx2).*A_t(:, midx2);
        nonlinear = Kerr + sum(Ra.*permute(A_t,[1 3 2]),3);
    else % polarized fields with an anisotropic Raman
        Rb = dt*fft(hbw.*ifft(Rb));

        nonlinear = Kerr + sum((Ra+Rb).*permute(A_t,[1 3 2]),3);
    end
else
    nonlinear = Kerr;
end
    
A_w = ifft(A_t); % a (N, num_modes) matrix
gain_rhs = (G.*A_w).'; % (num_modes, N) matrix, each column is a single w value
if ~sim.scalar
    gain_rhs = cat(3,gain_rhs(1:2:num_modes-1,:),gain_rhs(2:2:num_modes,:));
end
if sim.gpu_yes
    dA_dz_gain = permute(pagefun(@mtimes, transfer_matrix,gain_rhs),[2 1 3]);
else
    for polar_idx = 1:polar
        dA_dz_gain(:, :, polar_idx) = (transfer_matrix(:, :, polar_idx)*gain_rhs(:,:,polar_idx)).';
    end
end
% Put the separated polarization modes back into the same dimension of array.
if ~sim.scalar
    dA_dz_gain = cat(2,dA_dz_gain(:,:,1),dA_dz_gain(:,:,2));
    dA_dz_gain = dA_dz_gain(:,recovery_idx);
end

% Add nonlinearity and gain terms
dAdz = prefactor.*ifft(nonlinear) + dA_dz_gain;

end
