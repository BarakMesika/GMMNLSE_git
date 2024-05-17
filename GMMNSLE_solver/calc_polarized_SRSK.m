function varargout = calc_polarized_SRSK(mode_info,ellipticity,include_anisotropic_Raman)
%CALC_POLARIZED_SRSK It computes the SR,SK, when considering polarization
%modes, from "scalar SR values under orthogonal polarizations".
%
%   mode_info.nonzero_midx1234s - a (4,?) matrix
%   mode_info.SRa - a (num_nonzeros_midx1234s,number_frequency_points) array
%
%   ellipticity - the ellipticity of the polarization modes; Please refer to "Nonlinear Fiber Optics, eq (6.1.18) Agrawal" for the equations.
%                 0: linear polarization   -> (+,-)=(x,y)
%                 1: circular polarization -> (+,-)=(right,left)
%
%   include_anisotropic_Raman - true or false

mode_info.SRa = permute(mode_info.SRa,[1 3 2]); % put the frequency into the 3rd dimension
sSRa = size(mode_info.SRa);
if length(sSRa)==3
    Nf = sSRa(3); % the number of frequency points
else
    Nf = 1;
end
if sSRa(2) ~= 1
    mode_info.SRa = permute(mode_info.SRa,[2 1 3]); % SR needs to be a column vector
end
if sSRa(1) == 1 % only one spatial mode
    scalar_field = true;
else
    scalar_field = false;
end
if ~exist('include_anisotropic_Raman','var')
    include_anisotropic_Raman = false;
end
oddidx = @(x) 2*x-1;
evenidx = @(x) 2*x;

%% SRa: isotropic Raman term
odd12 = oddidx(mode_info.nonzero_midx1234s([1 2],:));
even12 = evenidx(mode_info.nonzero_midx1234s([1 2],:));
odd34 = oddidx(mode_info.nonzero_midx1234s([3 4],:));
even34 = evenidx(mode_info.nonzero_midx1234s([3 4],:));

SRa_midx = cat(2, [odd12; odd34],...
                  [odd12; even34],...
                  [even12; odd34],...
                  [even12; even34]);
SRa = repmat(mode_info.SRa,4,1,1);

SRa_info = mode_info;

% Sort SR indices
[sort_SRa_midx,sort_idx] = sortrows(SRa_midx.');
SRa_info.nonzero_midx1234s = sort_SRa_midx.';
SRa_info.SRa = SRa(sort_idx);

%% SK
odd1 = oddidx(mode_info.nonzero_midx1234s(1,:)); even1 = evenidx(mode_info.nonzero_midx1234s(1,:));
odd2 = oddidx(mode_info.nonzero_midx1234s(2,:)); even2 = evenidx(mode_info.nonzero_midx1234s(2,:));
odd3 = oddidx(mode_info.nonzero_midx1234s(3,:)); even3 = evenidx(mode_info.nonzero_midx1234s(3,:));
odd4 = oddidx(mode_info.nonzero_midx1234s(4,:)); even4 = evenidx(mode_info.nonzero_midx1234s(4,:));

switch ellipticity
    case 0 % linear polarizations
        odd1 = oddidx(mode_info.nonzero_midx1234s(1,:));         odd4 = oddidx(mode_info.nonzero_midx1234s(4,:));
        even1 = evenidx(mode_info.nonzero_midx1234s(1,:));       even4 = evenidx(mode_info.nonzero_midx1234s(4,:));
        odd23 = oddidx(mode_info.nonzero_midx1234s([2 3],:));
        even23 = evenidx(mode_info.nonzero_midx1234s([2 3],:));

        % Sk_midx = cat(2, [odd1;   odd23;  odd4],...
        %                  [odd1;  even23;  odd4],...
        %                  [even1;  odd23; even4],...
        %                  [even1; even23; even4]);
        % SK = 2/3*SRa + 1/3*Sk
        %
        SK_midx = cat(2, [odd12; odd34],...        % SR
                         [even12; even34],...      % SR
                         [odd12; even34],...       % 2/3*SR
                         [even12; odd34],...       % 2/3*SR
                         [odd1; even23; odd4],...  % 1/3*SR
                         [even1; odd23; even4]);   % 1/3*SR
        SK = mode_info.SRa.*[1 1 2/3 2/3 1/3 1/3];
    case 1 % circular polarization
        odd1 = oddidx(mode_info.nonzero_midx1234s(1,:)); even1 = evenidx(mode_info.nonzero_midx1234s(1,:));
        odd2 = oddidx(mode_info.nonzero_midx1234s(2,:)); even2 = evenidx(mode_info.nonzero_midx1234s(2,:));
        odd3 = oddidx(mode_info.nonzero_midx1234s(3,:)); even3 = evenidx(mode_info.nonzero_midx1234s(3,:));
        odd4 = oddidx(mode_info.nonzero_midx1234s(4,:)); even4 = evenidx(mode_info.nonzero_midx1234s(4,:));

        SK_midx = cat(2, [ odd1;  odd2;  odd3;  odd4],...  % 0
                         [ odd1;  odd2; even3; even4],...  % 1
                         [ odd1; even2;  odd3; even4],...  % 1
                         [even1;  odd2; even3;  odd4],...  % 1
                         [even1; even2;  odd3;  odd4],...  % 1
                         [even1; even2; even3; even4]);    % 0
        % SK = 2/3*SRa + 1/3*Sk
        Sk_term = [0 1 1 1 1 0];
        SRa_term = [1 1 0 0 1 1];
        SK = mode_info.SRa.*(2/3*SRa_term+1/3*Sk_term);
    otherwise % elliptical polarization
        % basis_o = (x+iry)/sqrt(1+r^2)
        % basis_e = (rx-iy)/sqrt(1+r^2)
        %
        % Notice that r=0 corresponds to basis_e = -iy. Since I separate the
        % linear polarization above, which has (basis_o = x, basis_e = y), it doesnt' matter here.
        %
        r = ellipticity; % match the notation with the "Nonlinear Fiber Optics, Agrawal"
        oo = (1-r^2)/(1+r^2);
        ee = -oo;
        oe = 2*r/(1+r^2);

        SK_midx = cat(2, [ odd1;  odd2;  odd3;  odd4],...  % (oo)^2
                         [ odd1;  odd2;  odd3; even4],...  % (oo)(oe)
                         [ odd1;  odd2; even3;  odd4],...  % ...
                         [ odd1; even2;  odd3;  odd4],...  % ...
                         [even1;  odd2;  odd3;  odd4],...  % ...
                         [ odd1;  odd2; even3; even4],...  % (oe)^2
                         [ odd1; even2;  odd3; even4],...  % ...
                         [even1;  odd2;  odd3; even4],...  % (oo)(ee)
                         [ odd1; even2; even3;  odd4],...  % ...
                         [even1;  odd2; even3;  odd4],...  % (oe)^2
                         [even1; even2;  odd3;  odd4],...  % ...
                         [ odd1; even2; even3; even4],...  % (ee)(oe)
                         [even1;  odd2; even3; even4],...  % ...
                         [even1; even2;  odd3; even4],...  % ...
                         [even1; even2; even3;  odd4],...  % ...
                         [even1; even2; even3; even4]);    % (ee)^2
        % SK = 2/3*SRa + 1/3*Sk
        Sk_term = [oo^2 ...
                   oo*oe oo*oe oo*oe oo*oe ...
                   oe^2 oe^2 ...
                   oo*ee oo*ee ...
                   oe^2 oe^2 ...
                   ee*oe ee*oe ee*oe ee*oe ...
                   ee^2];
        SRa_term = [1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1];
        SK = mode_info.SRa.*(2/3*SRa_term+1/3*Sk_term);
end

% Sort SK indices
[sort_SK_midx,sort_idx] = sortrows(SK_midx.');
SK_info.nonzero_midx1234s = sort_SK_midx.';
if Nf > 1
    SK = permute(SK,[3 1 2]);
    SK = SK(:,sort_idx);
    SK_info.SK = SK.';
else
    SK_info.SK = SK(sort_idx);
    
    if scalar_field % it needs to be a column vector
                    % If it's a scalar field, SRa is a scalar such that SK,
                    % calculated from SRa, is a row vector. Hence, SK needs a
                    % transpose.
        SK_info.SK = SK_info.SK.';
    end
end

%% SRb: anisotropic Raman term
if include_anisotropic_Raman
    switch ellipticity
        case 0 % linear polarizations

            % Srb_midx = cat(2, [ odd1;  odd2;  odd3;  odd4],...
            %                   [ odd1; even2;  odd3; even4],...
            %                   [even1;  odd2; even3;  odd4],...
            %                   [even1; even2; even3; even4]);
            %
            % Sk_midx = cat(2, [odd1;   odd23;  odd4],...
            %                  [odd1;  even23;  odd4],...
            %                  [even1;  odd23; even4],...
            %                  [even1; even23; even4]);
            %
            % SRb = 1/2*(Srb + Sk)
            %
            SRb_midx = cat(2, [ odd12;  odd34],...              % SR
                              [even12; even34],...              % SR
                              [ odd1; even2;  odd3; even4],...  % 1/2*SR
                              [even1;  odd2;  odd3; even4],...  % 1/2*SR
                              [ odd1; even2; even3;  odd4],...  % 1/2*SR
                              [even1;  odd2; even3;  odd4]);    % 1/2*SR
            SRb = mode_info.SRa.*[1 1 1/2 1/2 1/2 1/2];
        case 1 % circular polarization
            SRb_midx = cat(2, [ odd1;  odd2;  odd3;  odd4],...  % 1/2
                              [ odd1;  odd2; even3; even4],...  % 1/2
                              [ odd1; even2;  odd3; even4],...  % 1
                              [even1;  odd2; even3;  odd4],...  % 1
                              [even1; even2;  odd3;  odd4],...  % 1/2
                              [even1; even2; even3; even4]);    % 1/2
            % SRb = 1/2*(Srb + Sk)
            SRb = mode_info.SRa.*[1/2 1/2 1 1 1/2 1/2];
        otherwise % elliptical polarization
            % basis_o = (x+iry)/sqrt(1+r^2)
            % basis_e = (rx-iy)/sqrt(1+r^2)
            %
            % Notice that r=0 corresponds to basis_e = -iy. Since I separate the
            % linear polarization above, which has (basis_o = x, basis_e = y), it doesnt' matter here.
            %

            SRb_midx = SK_midx;
            
            % SRb = 1/2*(Srb + Sk)
            Srb_term = [1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1];
            SRb = mode_info.SRa.*(1/2*(Srb_term+Sk_term));
    end

    % Sort SRb indices
    [sort_SRb_midx,sort_idx] = sortrows(SRb_midx.');
    SRb_info.nonzero_midx1234s = sort_SRb_midx.';
    if Nf > 1
        SRb = permute(SRb,[3 1 2]);
        SRb = SRb(:,sort_idx);
        SRb_info.SRb = SRb.';
    else
        SRb_info.SRb = SRb(sort_idx); 
        if scalar_field % it needs to be a column vector
                        % If it's a scalar field, SRa is a scalar such that SRb,
                        % calculated from SRa, is a row vector. Hence, SRb needs a
                        % transpose.
            SRb_info.SRb = SRb_info.SRb.';
        end
    end
end


%% midx_34s: Part of Raman terms calculations under CPU
if isfield(mode_info,'nonzero_midx34s')
    SRa_midx34s = [odd34 even34];
    SRa_info.nonzero_midx34s = sortrows(SRa_midx34s.').';
    if include_anisotropic_Raman
        SRb_midx34s = cat(2, odd34,         ...
                            even34,         ...
                            [ odd3; even4], ...
                            [even3;  odd4]);
        SRb_info.nonzero_midx34s = sortrows(SRb_midx34s.').';
    end
end

%% Output
if include_anisotropic_Raman
    varargout = {SRa_info, SRb_info, SK_info};
else
    varargout = {SRa_info, SK_info};
end

end