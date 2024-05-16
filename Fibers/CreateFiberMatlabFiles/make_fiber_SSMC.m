c = 299792458;
lambda0 = 1030e-9;
lambdarange = 0e-9;
Nf = 1;

f0 = 2*pi*c/lambda0; 
frange = 2*pi*c/lambda0^2*lambdarange;
df = frange/Nf;
f = f0 + (-Nf/2:Nf/2-1)*df;
lambda = 2*pi*c./f; 
    
savedir = 'D:\GMMNLSE\Fibers\XLMA_YTF_100_400_480_cilindric_TMP';
fname = 'LP_';

NA = 0.11;
for ii=1:numel(lambda)
    fiber.ncl = refrIndex('silica', lambda(ii)*1e9);
    fiber.nco = sqrt(fiber.ncl^2+NA^2);
    fiber.a = 90e-6/2;
    modelist = fullfact([1 10]);
%     modelist(:,1) = modelist(:,1)-1; 
% %     modelist = [0 1];
    modelist = 'all';
    [modes, xx] = solve_STfiber(fiber, lambda(ii), 1024, 2*100e-6, modelist);
%     neff0(ii,:) = modes.neff(1:35);
%     neff1(ii,:) = modes.neff(36:end);
    for jj=1:size(modes.field,3)
        if ~exist(savedir, 'dir')
            mkdir(savedir)
        end
        phi = modes.field(:,:,jj);
        neff = modes.neff(jj);
        x = xx*1e6;
        wavelength = lambda(ii);
        imagesc(abs(phi).^2)
        pause(1)
%         ftmp = [fname num2str(modes.LP(jj,1),'%03.f') '-' num2str(modes.LP(jj,2),'%03.f') ...
%             '_wavelength' num2str(round(lambda(ii)*1e9),'%04.f')];
%         save(fullfile(savedir, ftmp), 'phi', 'neff', 'x', 'wavelength');
    end
end