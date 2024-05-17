function varargout = solve_gain_rate_eqn_linear_oscillator(sim,gain_rate_eqn,...
                                                           signal_fields,signal_fields_backward,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward,...
                                                           omegas,dt,deltaZ,...
                                                           cross_sections_pump,cross_sections,overlap_factor,N_total,FmFnN,GammaN)
%SOLVE_GAIN_RATE_EQN Solves the power of pump, ASE, and signal at
%z+-deltaZ, where +- depends on the propagation direction, forward or
%backward.
% It first solves N2, the ion density of the upper state, and uses this
% to calculate the power.
%

N = length(omegas); % the number of time/frequency points
time_window = N*(dt*1e-12); % unit: s
df = 1/(N*dt); % unit: THz
hbar = 6.62607004e-34/(2*pi);
E_photon = (hbar*1e12)*permute(omegas + 2*pi*sim.f0, [2 3 4 5 1]); % size: (1,1,1,1,N), unit: J

M = size(signal_fields,3); % the number of parallelization

if sim.scalar
    num_spatial_modes = size(signal_fields,2);
else % polarized fields
    num_spatial_modes = size(signal_fields,2)/2;
end

if sim.gpu_yes && sim.mpa_yes
    deltaZ = gpuArray(deltaZ);
end

E_forward  = signal_fields         *sqrt(time_window/gain_rate_eqn.t_rep);
E_backward = signal_fields_backward*sqrt(time_window/gain_rate_eqn.t_rep); % change it to the correct physical unit of W
                                                                   % AmAn*time_window = pulse energy
                                                                   % pulse energy/t_rep = W
                                                                   % t_rep: the time for a pulse to finish one round trip
Pdf_ASE_forward  = Power_ASE_forward *df;
Pdf_ASE_backward = Power_ASE_backward*df;

% AmAn field tensor
diag_idx = shiftdim(permute(0:num_spatial_modes^2*N:(num_spatial_modes^2*N*(M-1)),[1 3 4 2]) + permute(0:num_spatial_modes^2:(num_spatial_modes^2*(N-1)),[1 3 2]) + (1:(num_spatial_modes+1):num_spatial_modes^2)',-2); % find out the indices of the diagonal elements
if sim.scalar    
    AmAn_forward  = permute(E_forward ,[4 5 2 6 1 3]).*permute(conj(E_forward ),[4 5 6 2 1 3]); % size: (1,1,num_modes,num_modes,N,M)
    AmAn_backward = permute(E_backward,[4 5 2 6 1 3]).*permute(conj(E_backward),[4 5 6 2 1 3]); % size: (1,1,num_modes,num_modes,N,M)
else
    polarized_E_forward  = cat(4,E_forward(:,1:2:end-1,:),E_forward(:,2:2:end,:)); % separate the polarization modes
    AmAn_forward_p1  = permute(polarized_E_forward(:,:,:,1),[4 5 2 6 1 3]).*permute(conj(polarized_E_forward(:,:,:,1)),[4 5 6 2 1 3]);
    AmAn_forward_p2  = permute(polarized_E_forward(:,:,:,2),[4 5 2 6 1 3]).*permute(conj(polarized_E_forward(:,:,:,2)),[4 5 6 2 1 3]);
    AmAn_forward  = AmAn_forward_p1 + AmAn_forward_p2; % size: (1,1,num_modes,num_modes,N,M)
    polarized_E_backward = cat(4,E_backward(:,1:2:end-1,:),E_backward(:,2:2:end,:)); % separate the polarization modes
    AmAn_backward_p1 = permute(polarized_E_backward(:,:,:,1),[4 5 2 6 1 3]).*permute(conj(polarized_E_backward(:,:,:,1)),[4 5 6 2 1 3]);
    AmAn_backward_p2 = permute(polarized_E_backward(:,:,:,2),[4 5 2 6 1 3]).*permute(conj(polarized_E_backward(:,:,:,2)),[4 5 6 2 1 3]);
    AmAn_backward = AmAn_backward_p1 + AmAn_backward_p2; % size: (1,1,num_modes,num_modes,N,M)

    clear AmAn_forward_p1 AmAn_forward_p2 AmAn_backward_p1 AmAn_backward_p2;
end
AmAn = AmAn_forward + AmAn_backward;
clear AmAn_forward AmAn_backward
    
% -------------------------------------------------------------------------
% --------------------- Rate equation to get N2 ---------------------------
% -------------------------------------------------------------------------

% Ion density in the upper state
N2 = solve_N2( sim,gain_rate_eqn,N_total,E_photon,diag_idx,overlap_factor,cross_sections_pump,cross_sections,Power_pump_forward,Power_pump_backward,Pdf_ASE_forward,Pdf_ASE_backward,AmAn ); % unit: 1/um^3

% -------------------------------------------------------------------------
% -------------- Power equation to get Power_next_step --------------------
% -------------------------------------------------------------------------
if isequal(gain_rate_eqn.midx,1) % fundamental mode
    dx = [];
    A_core = pi*(gain_rate_eqn.core_diameter/2)^2;
else
    dx = gain_rate_eqn.mode_profile_dx; % unit: um
    A_core = [];
end

% -------------------------------------------------------------------------
% Power
% -------------------------------------------------------------------------
% Forward (at z+deltaZ):
Power_pump_forward     = solve_Power( 'pump',  sim.scalar,deltaZ*1e6,dx,A_core,overlap_factor.pump,  cross_sections_pump,N2,N_total,Power_pump_forward,        [], GammaN,    [],                        [],          []); % no spontaneous term for pump
if ~gain_rate_eqn.ignore_ASE
    Power_ASE_forward  = solve_Power( 'ASE',   sim.scalar,deltaZ*1e6,dx,A_core,overlap_factor.signal,cross_sections,     N2,N_total,Power_ASE_forward,   E_photon,     [], FmFnN, gain_rate_eqn.mode_volume,          []);
end

if sim.scalar
    field_input = permute(signal_fields,[4 5 2 6 1 3]);
else % polarized fields
    polarized_fields = cat(4,signal_fields(:,1:2:end-1,:),signal_fields(:,2:2:end,:)); % separate the polarization modes
    field_input = permute(polarized_fields,[5 6 2 7 1 3 4]);
end
[~,G]                  = solve_Power( 'signal',sim.scalar,deltaZ*1e6,dx,A_core,overlap_factor.signal,cross_sections,     N2,N_total,                 [],       [],     [], FmFnN,                        [], field_input); % no spontaneous term for signal
% -------------------------------------------------------------------------
% Backward (at z-deltaZ):
Power_pump_backward    = solve_Power( 'pump',  sim.scalar,-deltaZ*1e6,dx,A_core,overlap_factor.pump,  cross_sections_pump,N2,N_total,Power_pump_backward,      [], GammaN,    [],                        [],          []); % no spontaneous term for pump
% -------------------------------------------------------------------------

% Change the size back to (N,num_modes)
if sim.scalar
    G = permute(G,[5 3 6 1 2 4]);
else % polarized fields
    if isequal(gain_rate_eqn.midx,1) % fundamental mode
        G = permute(G,[5 3 6 1 2 4]);
    else
        % "recovery_idx" is used to put the separated polarization modes
        % back into the same dimension of array.
        recovery_idx = [(1:num_spatial_modes);(1:num_spatial_modes)+num_spatial_modes];
        recovery_idx = recovery_idx(:);
        G = cat(3,G(:,:,:,:,:,:,1),G(:,:,:,:,:,:,2));
        G = permute(G(:,:,recovery_idx,:,:,:),[5 3 6 1 2 4]);
    end
end

if gain_rate_eqn.export_N2
    varargout = {Power_pump_forward,Power_pump_backward,Power_ASE_forward,G,N2};
else
    varargout = {Power_pump_forward,Power_pump_backward,Power_ASE_forward,G};
end

end

%%
function N2 = solve_N2( sim,gain_rate_eqn,N_total,E_photon,diag_idx,overlap_factor,cross_sections_pump,cross_sections,Power_pump_forward,Power_pump_backward,Power_ASE_forward,Power_ASE_backward,AmAn )
%N2 solves the ion density in the upper state under the steady state of the
%rate equation
%
% computational dimension here: (Nx,Nx,num_modes,N,M), M: parallelization in MPA

hbar = 6.62607004e-34/(2*pi);
c = 299792458;

% Rate/photon_energy:
% size: (Nx,Nx,num_modes,1,N), unit: W
% pump
sN = size(N_total);
sP = size(Power_pump_forward);
if sim.gpu_yes
    R_over_photon = struct('pump_forward_absorption' ,zeros([sN,sP(3:end)],'gpuArray'),'pump_forward_emission', zeros([sN,sP(3:end)],'gpuArray'),...
                           'pump_backward_absorption',zeros([sN,sP(3:end)],'gpuArray'),'pump_backward_emission',zeros([sN,sP(3:end)],'gpuArray'));
else
    R_over_photon = struct('pump_forward_absorption' ,zeros([sN,sP(3:end)]),'pump_forward_emission', zeros([sN,sP(3:end)]),...
                           'pump_backward_absorption',zeros([sN,sP(3:end)]),'pump_backward_emission',zeros([sN,sP(3:end)]));
end
if any(strcmp(gain_rate_eqn.pump_direction,{'co','bi'}))
    R_over_photon.pump_forward_absorption  = overlap_factor.pump*cross_sections_pump.absorption.*Power_pump_forward /hbar*(gain_rate_eqn.pump_wavelength*1e-9)/(2*pi*c);
    R_over_photon.pump_forward_emission    = overlap_factor.pump*cross_sections_pump.emission.*  Power_pump_forward /hbar*(gain_rate_eqn.pump_wavelength*1e-9)/(2*pi*c);
end
if any(strcmp(gain_rate_eqn.pump_direction,{'counter','bi'}))
    R_over_photon.pump_backward_absorption = overlap_factor.pump*cross_sections_pump.absorption.*Power_pump_backward/hbar*(gain_rate_eqn.pump_wavelength*1e-9)/(2*pi*c);
    R_over_photon.pump_backward_emission   = overlap_factor.pump*cross_sections_pump.emission.*  Power_pump_backward/hbar*(gain_rate_eqn.pump_wavelength*1e-9)/(2*pi*c);
end

% ASE and signal
if ~gain_rate_eqn.ignore_ASE
    if sim.scalar
        Power_ASE = Power_ASE_forward + Power_ASE_backward;
    else % polarized fields
        Power_ASE = Power_ASE_forward(:,:,1:2:end-1,:,:,:)  + Power_ASE_forward(:,:,2:2:end,:,:,:) + ... % forward
                    Power_ASE_backward(:,:,1:2:end-1,:,:,:) + Power_ASE_backward(:,:,2:2:end,:,:,:);     % backward
    end
    AmAn(diag_idx) = AmAn(diag_idx) + Power_ASE; % AmAn(:,:,n,n,:,:) = AmAn(:,:,n,n,:,:) + Power_ASE(:,:,n,:,:,:)
end

% "trapz" and "fftshift" are slower on GPU than on CPU when the matrix is
% not large, so for slightly multimode case, it can still be faster for
% CPU.
% Because "fftshift" can be 200x slower for my own PC, I write my own function, "mytrapzfftshift5".
% 
% I later find out using sum(x,5) can be 10x faster than "mytrapzfftshift5" despite negligible inaccuracy.
%
% Original code:
%   absorption_integral_mn = trapz(fftshift(AmAn.*(cross_sections.absorption./E_photon),5),5); % size: (1,1,num_modes,num_modes,1,M)
%   emission_integral_mn   = trapz(fftshift(AmAn.*(cross_sections.emission  ./E_photon),5),5);
% Modified code:
%   absorption_integral_mn = mytrapzfftshift5(AmAn.*(cross_sections.absorption./E_photon)); % size: (1,1,num_modes,num_modes,1,M)
%   emission_integral_mn   = mytrapzfftshift5(AmAn.*(cross_sections.emission  ./E_photon));
absorption_integral_mn = sum(AmAn.*(cross_sections.absorption./E_photon),5); % size: (1,1,num_modes,num_modes,1,M)
emission_integral_mn   = sum(AmAn.*(cross_sections.emission  ./E_photon),5);

% For SMF (fundamental mode only), the computations below all have the
% length 1 or M, and thus can be faster with CPU, instead of GPU.
if sim.gpu_yes && isequal(gain_rate_eqn.midx,1) % fundamental mode
    absorption_integral_mn = gather(absorption_integral_mn);
    emission_integral_mn   = gather(emission_integral_mn);
end

% It's real! Use "real" to save the memory.
R_over_photon.ASE_signal_absorption = real(sum(sum(absorption_integral_mn.*overlap_factor.signal,3),4)); % size: (Nx,Nx,1,1,1,M)
R_over_photon.ASE_signal_emission   = real(sum(sum(  emission_integral_mn.*overlap_factor.signal,3),4));

% ion density in the upper state
% N2 = N_total*sum(all the absorption terms)/
%      ( sum(all the absorption and emission terms) + spontaneous emission term )
total_absorption = (R_over_photon.pump_forward_absorption + R_over_photon.pump_backward_absorption) + ... % pump
                    R_over_photon.ASE_signal_absorption;                                                  % ASE, signal
total_emission   = (R_over_photon.pump_forward_emission   + R_over_photon.pump_backward_emission) + ... % pump
                    R_over_photon.ASE_signal_emission;                                                  % ASE, signal
N2 = N_total.*total_absorption./...
     (total_absorption + total_emission + ... % absorption and emission terms
      1/gain_rate_eqn.tau);                   % spontaneous emission

end

%%
function [Pnext,signal_out] = solve_Power( field_type,isscalar,deltaZ,dx,A_core,overlap_factor,cross_sections,N2,N_total,P0,E_photon,GammaN,FmFnN,mode_volume,Am )
%SOLVE_POWER solves Power(z+deltaZ) for pump, ASE, and signal.
%
%   deltaZ: um
%

signal_out = [];

if isempty(dx) % fundamental mode
    overlap_factor = overlap_factor*A_core; % For the fundamental mode, the integral w.r.t. x and y can be done first, which becomes overlap_factor here.
    
    % spontaneous emission
    if isequal(field_type,'ASE') % ASE
        % Put "E_photon" at the end, otherwise, the result will be all zeros
        % when multiplying from left to right under "single-precision".
        dPdz_spontaneous = overlap_factor*cross_sections.emission*N2.*(E_photon*1e12); % unit: W/THz/um
                                                                                     % 1e12 is to transform Hz into THz (J=W/Hz)
    else % signal: ignore spontaneous term, pump: no spontaneous term
        dPdz_spontaneous = 0;
    end
    switch field_type
        case 'signal' % amplification factor
            signal_out = sqrt( 1 + overlap_factor*((cross_sections.emission + cross_sections.absorption)*N2 - cross_sections.absorption*N_total).*deltaZ );
        otherwise
            fz = ( overlap_factor*((cross_sections.emission + cross_sections.absorption)*N2 - cross_sections.absorption*N_total ).*P0 + dPdz_spontaneous ).*deltaZ; % unit: W/THz for ASE, W for pump
    end
    
else % multimode
    %trapz2 = @(x) trapz(trapz(x,1),2)*dx^2; % take the integral w.r.t. the x-y plane
    trapz2 = @(x) sum(sum(x,1),2)*dx^2; % take the integral w.r.t. the x-y plane
    
    cross_section_all = cross_sections.emission + cross_sections.absorption;
    
    switch field_type
        case 'signal' % ignore spontaneous term
            FmFnN2 = trapz2(overlap_factor.*N2);
            
            % Calculate gA*deltaZ after passing through the gain.
            signal_out = permute( sum( deltaZ/2.*(cross_section_all.*FmFnN2 - cross_sections.absorption.*FmFnN).*Am ,3) ,[1 2 4 3 5 6 7]); % Am_after_gain - Am
        case 'ASE'
            num_modes = size(P0,3);
            diag_idx = sub2ind([num_modes num_modes],1:num_modes,1:num_modes);
            GammaN2 = trapz2(overlap_factor(:,:,diag_idx).*N2); % overlap_factor*N2
            GammaN = FmFnN(:,:,diag_idx); % overlap_factor*N_total
            
            if ~isscalar % polarized fields
                num_spatial_modes = size(P0,3)/2;
                P0 = cat(7,P0(:,:,1:2:end-1,:,:,:),P0(:,:,2:2:end,:,:,:));
            end
            
            % Put "E_photon" at the end, otherwise, the result will be all 
            % zeros when multiplying from left to right under "single" 
            % precision.
            fz = real( GammaN2.*(cross_section_all.*P0 + cross_sections.emission.*E_photon*1e12/mode_volume) - GammaN.*cross_sections.absorption.*P0 ).*deltaZ; % W/THz
                                                                                                                                                                % 1e12 in the spontaneous term is to transform Hz into THz
            
            if ~isscalar % polarized fields
                % "recovery_idx" is used to put the separated polarization modes
                % back into the same dimension of array.
                recovery_idx = [(1:num_spatial_modes);(1:num_spatial_modes)+num_spatial_modes];
                recovery_idx = recovery_idx(:);
            end
        case 'pump'
            GammaN2 = trapz2(overlap_factor.*N2);
            fz = ( GammaN2*cross_section_all - GammaN*cross_sections.absorption ).*P0.*deltaZ; % W
    end
end

if isequal(field_type,'signal')
    Pnext = [];
else
    if isempty(dx) % fundamental mode
        Pnext = P0 + fz;
    else % multimode
        fz(:,:,:,:,:,1) = fz(:,:,:,:,:,1)/2;
        if size(P0) == 1
            Pnext = P0*ones(size(fz));
            gpu_yes = isequal(class(P0),'gpuArray');
            if gpu_yes
                single_yes = isequal(classUnderlying(P0),'single');
            else
                single_yes = isequal(class(P0),'single');
            end
            if single_yes
                Pnext = single(Pnext);
            end
            if gpu_yes
                Pnext = gpuArray(Pnext);
            end
        else
            Pnext = P0;
        end
        Pnext(:,:,:,:,:,2:end) = P0(:,:,:,:,:,1) + cumsum(fz(:,:,:,:,:,1:end-1),6) + fz(:,:,:,:,:,2:end)/2;
        Pnext(Pnext<0) = 0; % remove artificial negative values because of backward-computation
    end
    
    if size(P0,7) == 2 % ASE under polarized multimode
        Pnext = cat(3,Pnext(:,:,:,:,:,:,1),Pnext(:,:,:,:,:,:,2));
        Pnext = Pnext(:,:,recovery_idx,:,:,:);
    end
end

end

%%
function y = mytrapzfftshift5( x )
%MYTRAPZFFTSHIFT5
%   "trapz" and "fftshift" are slower on GPU than on CPU when the matrix is
%not large, so for slightly multimode case, it can still be faster for
%CPU.
%Because "fftshift" can be 200x slower for my own PC, I write my own function.
%
% This is used to replace "trapz(fftshift(x,5),5)".

sx = ceil(size(x,5)/2);

y = trapz(cat(5,x(:,:,:,:,sx+1:end,:,:),x(:,:,:,:,1:sx,:,:)),5);

end