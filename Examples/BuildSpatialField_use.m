
%% with real simulation
if ~isfield(others, 'Nx')
    others.Nx = 1024;
end
total_field = BuildSpatialField(output_field.fields(:,:,end),fiber, sim, others);

% make integral on all time ~20ps
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
imagesc(x, x, abs(time_integral).^2);

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
folder = 'GIF625_KBSC_10modes_1030nm_energy_sweep';
cd(folder);
mfiles = ls('*.mat');
cd('..');

energy_length = length(mfiles);

figure;
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
    title(['Energy = ' num2str(input_field.E_tot * 1e-3) 'nJ' ]); 

    % get Spatial Spectral
    

end

%% 

% the spectrum at each point on the grid -> spatial_spectral(X,Y,lambda) 
spatial_spectal = ifft(total_field,[],3);
spatial_spectal = fftshift(spatial_spectal , 3);
spatial_spectal = abs(spatial_spectal).^2;

%%

teta = 0:0.01:2*pi;
corex = fiber.radius*sin(teta);
corey = fiber.radius*cos(teta);
dx = fiber.radius*3/others.Nx; % um
x =  (-others.Nx/2:others.Nx/2-1)*dx;

idx = linspace(71,1,8);
figure;
for i = 1:length(idx)

    subplot(4,4,i);
    imagesc(x, x, spatial_spectal(:,:,idx(i)));
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


%%
d_lambda = abs(others.lambda(2) - others.lambda(1));
spectral_integral = sum(spatial_spectal, 3) * d_lambda;

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