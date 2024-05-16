function [modes_struc, x] = solve_STfiber(fiber, lambda0, Nx, Wx, modeList)
assert(isfield(fiber,'ncl'),'n cladding not found. Structure fiber must have ''ncl'' field');
assert(isfield(fiber,'nco'),'n core not found. Structure fiber must have ''nco'' field');
assert(isfield(fiber,'a'),'core radius not found. Structure fiber must have ''a'' field');
assert(fiber.nco>fiber.ncl, 'nco must be larger than ncl');

a = fiber.a;
ncl = fiber.ncl;
nco = fiber.nco;
NA = sqrt(nco^2 - ncl^2);

%spatial grid and number of modes
dx = Wx/Nx; % um
x = (-Nx/2:Nx/2-1)*dx;
[X, Y] = meshgrid(x, x);
V = 2*pi*a/lambda0*NA;
Mnumber = 4/pi^2*V^2;
fprintf('Expected %d modes\n', Mnumber);
% Check modeList
if isnumeric(modeList)
    secondDim = size(modeList,2);
    assert((secondDim==2), 'Incorect mode list, should be k by 2 array');
    flag.mode = 'list';
    flag.val = 1;
    fprintf('Solving for listed modes\n');
else
    assert(strcmpi(modeList, 'all'), 'Incorect mode list, should be ''all'' or k by 2 array');
    flag.mode = 'all';
    flag.val = 1;
    fprintf('Solving for all modes\n');
end

% find l to start with
if strcmpi(modeList, 'all')
    l = 0;
else
    l = min(modeList(:,1));
end

% Solve for mode LP_l_m
[theta,rho] = cart2pol(X,Y);
z = 0.001:0.001:V;
dz = z(2)-z(1);
modes_struc = struct;
runningInd = 1;
while flag.val
    fLeft = @(z) z.*besselj(l+1,z)./besselj(l,z);
    fRight = @(z) sqrt(V^2-z.^2).*besselk(l+1,sqrt(V^2-z.^2))./besselk(l,sqrt(V^2-z.^2));
    rhs = feval(fRight, z);
    lhs = feval(fLeft, z);
    signFlip = sign(lhs-rhs);
    signFlip = diff(signFlip)/2;
    signFlip = [0 signFlip];
    signFlip(signFlip<0.5)=0;
    zposition = find(signFlip);
    
    % if there is no sign flip stop
    if sum(signFlip)==0
        fprintf('%d modes are found, done \n', runningInd-1)
        break;
    end
    % find m ind for given l
    if isnumeric(modeList)
        mind = modeList(modeList(:,1)==l,2);
        if max(mind)>sum(signFlip)
            fprintf('m%d l%d does not exist\n Change m\n', max(mind), l);
            break;
        end
    else
        mind = 1:sum(signFlip);
    end
    
    for ni=1:numel(mind)
        ii=mind(ni);
        zrange = [z(zposition(ii))-dz, min([z(zposition(ii))+dz max(z)])];
        tmp = fzero(@(z)fLeft(z)-fRight(z),zrange);
        modes_struc.Solution(runningInd) = tmp;
        modes_struc.LP(runningInd,:) = [l ii];
        kT = tmp./a;
        gamma = sqrt(V^2-tmp.^2)./a;
        modes_struc.betta(runningInd) = sqrt(nco^2*(2*pi/lambda0)^2-kT.^2);
        modes_struc.neff(runningInd) = modes_struc.betta(runningInd)*lambda0/2/pi;
       
        tmpJ = besselj(l, kT*rho).*double(rho<=a);
        tmpK = besselj(l, kT*a)*besselk(l, gamma*rho)/besselk(l, gamma*a).*double(rho>a);
        tmpK(isnan(tmpK))=0; % correct besselk % rho=0
        tmpJK = tmpJ+tmpK;
        modes_struc.field(:,:,runningInd) = tmpJK.*cos(l*theta);
        tmp = modes_struc.field(:,:,runningInd);
        modes_struc.Aeff(runningInd) = (sum(sum(abs(tmp).^2))*dx*dx).^2/(sum(sum(abs(tmp).^4))*dx*dx);
        modes_struc.PinCore(runningInd) = (sum(sum(abs(tmp.*double(rho<=a)).^2))*dx*dx)/(sum(sum(abs(tmp).^2))*dx*dx);
        fprintf('LP:%d %d\tbetta:%f /um\tn_eff:%f\tA_eff:%f um^2\tPinCore:%f%%\n',...
            l,ii,modes_struc.betta(runningInd)*1e-6, modes_struc.neff(runningInd),...
            modes_struc.Aeff(runningInd)*1e12, modes_struc.PinCore(runningInd)*100);
        runningInd = runningInd+1;
    end
    % Set new l
    if isnumeric(modeList)
        if max(modeList(:,1))==l
            fprintf('%d modes are found, done \n', runningInd-1)
            break;
        else
            tmp = modeList(modeList(:,1)>l,1);
            l = tmp(1);
        end
    else
        l = l+1;
    end
end

