

%% Temp

z_idx = 61;

if ~isfield(others, 'Nx')
    others.Nx = 1024;
end

total_field = BuildSpatialField(output_field.fields(:,:,z_idx),fiber, sim, others);

% the spectrum at each point on the grid -> spatial_spectral(X,Y,lambda) 
spatial_spectral = ifft(total_field,[],3);
spatial_spectral = fftshift(spatial_spectral , 3);
spatial_spectral = abs(spatial_spectral).^2;
clear total_field;

%% with real simulation
if ~isfield(others, 'Nx')
    others.Nx = 1024;
end
total_field = BuildSpatialField(output_field.fields(:,:,1),fiber, sim, others);

% make integral on all time 
time_integral = sum(abs(total_field).^2, 3) * input_field.dt;
% energy_tot = sum(sum(time_integral)) % check that we get all of the energy



%% plot the spatial intensity
teta = 0:0.01:2*pi;
corex = fiber.radius*sin(teta);
corey = fiber.radius*cos(teta);

dx = fiber.radius*3/others.Nx; % um
x =  (-others.Nx/2:others.Nx/2-1)*dx;

gg=figure('Position',[1 1 600 600]);
% clims = [1e1 max(max(abs(time_integral).^2))];
imagesc(x, x, time_integral);

colorbar;
xlim([min(x) max(x)]);ylim([min(x) max(x)]);
colormap(jet(128))
axis square
hold on
plot(corex, corey, 'LineWidth', 1, 'Color', 'y', 'LineStyle', ':')
hold off
title('Output Intensity'); 
        


%% plot the spatial intensity for all of the files

% load folder
folder = 'GIF625_KBSC_TC_10modes_1030nm_energy_sweep';
cd(folder);
mfiles = ls('*.mat');
cd('..');

energy_length = length(mfiles);

figure;
sgtitle('DC ; 5ps FWHM ; 10modes ; modes 1-5 exited equally')
for i=1:energy_length

    % load file
    load([folder '\' mfiles(i,:)]);

    % fix Nx issue
    if ~isfield(others, 'Nx')
        others.Nx = 1024;
    end

    total_field = BuildSpatialField(output_field.fields(:,:,end),fiber, sim, others);
    time_integral = sum(abs(total_field).^2, 3) * input_field.dt;

    % plot the CCD beam

    teta = 0:0.01:2*pi;
    corex = fiber.radius*sin(teta);
    corey = fiber.radius*cos(teta);
    dx = fiber.radius*3/others.Nx; % um
    x =  (-others.Nx/2:others.Nx/2-1)*dx;

    subplot(3,5,i)
    imagesc(x, x, abs(time_integral).^2); colorbar;
    xlim([min(x) max(x)]);ylim([min(x) max(x)]);
    colormap(jet(128))
    axis square
    hold on
    plot(corex, corey, 'LineWidth', 1, 'Color', 'y', 'LineStyle', ':')
    hold off
    title(['Energy = ' num2str(input_field.E_tot * 1e-3) 'nJ ;' ...
          ' Peak Power = ' num2str(max(abs(input_field).^2)) ]); 

    

end

%% plot Spatial Spectrum

% total_field = BuildSpatialField(output_field.fields(:,:,end),fiber, sim, others);

% the spectrum at each point on the grid -> spatial_spectral(X,Y,lambda) 
spatial_spectal = ifft(total_field,[],3);
spatial_spectal = fftshift(spatial_spectal , 3);
spatial_spectal = abs(spatial_spectal).^2;

%% spectrum at a spesific point

figure;
plot(fftshift(others.lambda), squeeze(spatial_spectal(132,135,:)));
xlabel('Wavelength [nm]')
ylabel('Spectral Power [a.u]')
title('Spectrum at the center of the beam')
xlim([870 1200])
grid minor;

%%

teta = 0:0.01:2*pi;
corex = fiber.radius*sin(teta);
corey = fiber.radius*cos(teta);
dx = fiber.radius*3/others.Nx; % um
x =  (-others.Nx/2:others.Nx/2-1)*dx;

idx = linspace(71,1,8);
figure;
sgtitle(['UC ; Enrgy= ' num2str(input_field.E_tot * 1e-3) 'nJ']);

for i = 1:length(idx)

    subplot(4,4,i);
    imagesc(x, x, spatial_spectal(:,:,idx(i)) ) ;
    xlim([min(x) max(x)]);ylim([min(x) max(x)]);
    colormap(jet(128))
    axis square
    hold on
    plot(corex, corey, 'LineWidth', 1, 'Color', 'y', 'LineStyle', ':')
    hold off
    title([num2str(others.lambda(idx(i))) 'nm']);

    subplot(4,4,17-i);
    imagesc( x, x, spatial_spectal(:,:,end-idx(i)) ); 
    xlim([min(x) max(x)]);ylim([min(x) max(x)]);
    colormap(jet(128))
    axis square
    hold on
    plot(corex, corey, 'LineWidth', 1, 'Color', 'y', 'LineStyle', ':')
    hold off
    title([num2str(others.lambda(end-idx(i))) 'nm']);

end


%% Intensity figure fron the Field vs. Spectrum

d_lambda = abs(others.lambda(2) - others.lambda(1));
spectral_integral = sum(abs(spatial_spectal).^2, 3) * d_lambda;

subplot(1,2,1)
imagesc(x, x, spectral_integral); colorbar;
xlim([min(x) max(x)]);ylim([min(x) max(x)]);
colormap(jet(128))
axis square
hold on
plot(corex, corey, 'LineWidth', 1, 'Color', 'y', 'LineStyle', ':')
hold off
title('Total Spectal Power');

subplot(1,2,2)
imagesc(x, x, time_integral);
colorbar;
xlim([min(x) max(x)]);ylim([min(x) max(x)]);
colormap(jet(128))
axis square
hold on
plot(corex, corey, 'LineWidth', 1, 'Color', 'y', 'LineStyle', ':')
hold off
title('Output Intensity'); 


%% Parseval's theorem
% Ssum = sum(sum(spectral_integral)) * d_lambda;
% Tsum = sum(sum(time_integral)) * input_field.dt;
% Tsum/Ssum 