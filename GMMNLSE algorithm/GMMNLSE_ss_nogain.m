function A1 = GMMNLSE_ss_nogain(A0, dt, sim, prefactor, SRa_info, SRb_info, SK_info, D, haw, hbw)
%GMMNLSE_SS_NOGAIN Take one step according to the GMMNLSE, using the split step method without gain
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
%
% Output:
% A1 - (N, num_modes) matrix with the field after the step, for each mode, in the frequency domain

N = size(A0, 1); % number of time/freq points
num_modes = size(A0, 2);

anisotropic_Raman_included = ~sim.scalar & sim.Raman_model==2;

% Setup the matrices
if sim.gpu_yes
    if sim.single_yes
        Kerr = complex(zeros(N, num_modes, 'single', 'gpuArray'));
        Ra = complex(zeros(N, num_modes, num_modes, 'single', 'gpuArray'));
    else
        Kerr = complex(zeros(N, num_modes, 'gpuArray'));
        Ra = complex(zeros(N, num_modes, num_modes, 'gpuArray'));
    end
else
    if sim.single_yes
        Kerr = complex(zeros(N, num_modes, 'single'));
        Ra = complex(zeros(N, num_modes, num_modes, 'single'));
    else
        Kerr = complex(zeros(N, num_modes));
        Ra = complex(zeros(N, num_modes, num_modes));
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

expD = exp(D);

% 1) Propagate through the first dispersion section
A_t = fft(expD.*A0);

% 2) Propagate through the nonlinearity with RK4
k1 = N_op(A_t,      dt, sim, SK_info, SRa_info, SRb_info, Kerr, Ra, Rb, haw, hbw, anisotropic_Raman_included, prefactor, N, num_modes)*sim.deltaZ;
k2 = N_op(A_t+k1/2, dt, sim, SK_info, SRa_info, SRb_info, Kerr, Ra, Rb, haw, hbw, anisotropic_Raman_included, prefactor, N, num_modes)*sim.deltaZ;
k3 = N_op(A_t+k2/2, dt, sim, SK_info, SRa_info, SRb_info, Kerr, Ra, Rb, haw, hbw, anisotropic_Raman_included, prefactor, N, num_modes)*sim.deltaZ;
k4 = N_op(A_t+k3,   dt, sim, SK_info, SRa_info, SRb_info, Kerr, Ra, Rb, haw, hbw, anisotropic_Raman_included, prefactor, N, num_modes)*sim.deltaZ;

A_w_afternonlin = ifft(A_t + (k1+2*(k2+k3)+k4)/6);

% 3) Propagate through the second dispersion section
A1 = expD.*A_w_afternonlin;

end

function dAdz = N_op(A_t, dt, sim,...
                     SK_info, SRa_info, SRb_info,...
                     Kerr, Ra, Rb,...
                     haw, hbw, anisotropic_Raman_included,...
                     prefactor,...
                     N, num_modes)
%N_op Calculate dAdz

% Calculate the large num_modes^4 sum term
if sim.gpu_yes
    % If using the GPU, do the computation with fast CUDA code
    if isempty(SK_info) % scalar fields
        if sim.Raman_model==0
            Kerr = feval(sim.kernel, Kerr, complex(A_t), SRa_info.SRa, SRa_info.nonzero_midx1234s,  N, 1, sim.SK_factor, size(SRa_info.nonzero_midx1234s, 2), num_modes);
        else
            [Kerr, Ra] = feval(sim.kernel, Kerr, Ra, complex(A_t), SRa_info.SRa, SRa_info.nonzero_midx1234s,  N, 1, sim.SK_factor, size(SRa_info.nonzero_midx1234s, 2), num_modes);
        end
    else % polarized fields
        switch sim.Raman_model
            case 0
                Kerr = feval(sim.kernel, Kerr, complex(A_t), SK_info.SK, SK_info.nonzero_midx1234s, N, 1, size(SK_info.nonzero_midx1234s, 2), num_modes);
            case 1
                [Kerr, Ra] = feval(sim.kernel, Kerr, Ra, complex(A_t), SRa_info.SRa, SRa_info.nonzero_midx1234s, SK_info.SK, SK_info.nonzero_midx1234s, N, 1, size(SRa_info.nonzero_midx1234s, 2), size(SK_info.nonzero_midx1234s, 2), num_modes);
            case 2
                [Kerr, Ra, Rb] = feval(sim.kernel, Kerr, Ra, Rb, complex(A_t), SRa_info.SRa, SRa_info.nonzero_midx1234s, SRb_info.SRb, SRb_info.nonzero_midx1234s, SK_info.SK, SK_info.nonzero_midx1234s, N, 1, size(SRa_info.nonzero_midx1234s, 2), size(SK_info.nonzero_midx1234s, 2), num_modes);
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
    %        for nz_idx = 1:size(SRa_info.nonzero_midx34s, 2)
    %            midx3 = SRa_info.nonzero_midx34s(1, nz_idx);
    %            midx4 = SRa_info.nonzero_midx34s(2, nz_idx);
    %            Ra_mn(:, midx3, midx4) = A_t(:, midx3).*conj(A_t(:, midx4));
    %        end
    %    
    %    Then calculate the num_modes^4 sum
    %        for nz_idx = 1:size(SRa_info.nonzero_midx1234s, 2)
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
end

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

% Now eveyerthing has been summed into Kerr, so transform into the
% frequency domain for the prefactor, then back into the time domain
dAdz = fft(prefactor.*ifft(nonlinear));

end