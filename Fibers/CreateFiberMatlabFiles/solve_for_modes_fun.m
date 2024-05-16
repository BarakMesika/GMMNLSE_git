function solve_for_modes_fun(data)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script builds the fiber index profile and calls the svmodes function
% to solve for the lowest m modes over a range of frequencies.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set parameters
Nf = data.Nf;                                     % number of frequency points at which the modes will be calculated
lambda0 = data.lambda0;                          % center wavelength [m]
lrange = data.lrange;                                 % wavelength range [m].
                                            % If 0 only the center wavelength will be used
num_modes = data.num_modes;                             % number of modes to compute
radius = data.radius;                    % outer radius of fiber [um]
folder_name = data.folder_name;    % folder where the output will be stored

Nx = data.Nx;                                  % number of spatial grid points
spatial_window = data.spatial_window;                        % full spatial window size, in um
profile_function = data.profile_function;             % function that builds the fiber
extra_params.ncore_diff = data.extra_params.ncore_diff;           % difference between the index at the center of the core, and the cladding
extra_params.alpha = data.extra_params.alpha;                  % Shape parameter
extra_params.NA = data.extra_params.NA;                     % Numerical Aperture
%% Calculate the modes

mkdir(folder_name);
if ispc
    sep_char = '/';
else
    sep_char = '\';
end
    
% Set the range in frequency space, which is more objective
c = 2.99792458e-4; % speed of ligth m/ps
if lrange == 0
    l = lambda0*10^6;
else
    f0 = c/lambda0; % center frequency in THz
    frange = c/lambda0^2*lrange;
    df = frange/Nf;
    f = f0 + (-Nf/2:Nf/2-1)*df;
    l = c./f*10^6; % um


end

% At each wavelength, calculate the modes
for kk = 1:length(l)
    lambda = l(kk); % wavelength
    
    % Build the index profile. The funcation can be arbitrary, and can take
    % any extra parameters
    [epsilon, x, dx] = profile_function(lambda, Nx, spatial_window, radius, extra_params);
    guess = sqrt(epsilon(Nx/2, Nx/2));

    % Quickly show the index profile to make sure everything's working
    % correctly
    gg=figure;
    subplot(2,1,1)

    pcolor(x,x,epsilon.^0.5)

    colormap(gray)
    colormap(flipud(colormap))
    shading interp
    axis square

    subplot(2,1,2)
    plot(x,epsilon(:,Nx/2).^0.5)
    grid on

    saveas(gg,[folder_name sep_char 'fiberprofile'],'fig');
    print(gg,[folder_name sep_char 'fiberprofile'],'-dpng');
    close (gg)

    % Actually do the calculation
    field = 'scalar'; % See svmodes for details
    boundary = '0000'; % See svmodes for details
    t_justsolve = tic();
    [phi1,neff1]=svmodes(lambda,guess,num_modes,dx,dx,epsilon,boundary,field);
    disp(['Wavelength no. ' num2str(kk)]);
    toc(t_justsolve);

    % Save each mode in a separate file
    teta = 0:0.01:2*pi;
    corex = radius*sin(teta);
    corey = radius*cos(teta);
%     load('colormapYAZ', 'cmap');
%     cmap = hsv(255);
    dx = x(2)-x(1);
    for ii=1:num_modes
        phi = phi1(:,:,ii);
        neff = neff1(ii);
        
        Aeff(ii) = (sum(sum(abs(phi).^2))*dx*dx).^2/(sum(sum(abs(phi).^4))*dx*dx);
        fprintf('Mode#: %d\tn_eff: %f\tA_eff: %f um^2 \n',ii,neff,Aeff(ii));
        
        gg=figure('Position',[1 1 600 600]);
        imagesc(x, x, abs(phi).^2)
        xlim([min(x) max(x)]);ylim([min(x) max(x)]);
%         shading interp
        colormap(jet(128))
        axis square
        hold on
        plot(corex, corey, 'LineWidth', 1, 'Color', 'y', 'LineStyle', ':')
        hold off
        title(['n_{eff} = ' num2str(neff) '      ' 'Aeff = ' num2str(Aeff(ii))]) 
        
        % Save the file with identifying information
        % fname=[folder_name sep_char 'radius'  num2str(round(radius)) 'boundary' boundary 'field'...
        %     field  'mode' num2str(ii,'%03.f') 'wavelength' num2str(round(lambda*1000),'%04.f')];
        
        % added by BarakM 11.5.24
        fname=[folder_name sep_char 'radius'  strrep(num2str(radius), '.', '_') 'boundary' boundary 'field'...
            field  'mode' num2str(ii,'%03.f') 'wavelength' strrep(num2str(lambda*1000), '.', '_')];

        print(gg,fname,'-dpng')
        save(fname,'x','phi','epsilon','neff')
        close(gg)
        
    end
end
beep;
end