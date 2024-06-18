clear; clc; close;

%% run TL, UC and DC

% TL
TL_title_name = 'TL ; 175fs FWHM ; 10modes ; modes 1-5 exited equally';
folder_TL = 'GIF625_KBSC_TL_10modes_1030nm_energy_sweep';
fiber = @TL_KBSC_10modes_1030nm;
energy = linspace(0.5e3, 18e3, 10);
input_energy_sweep(fiber, energy);

file_path = [folder_TL '\data_004.mat'];
plot_spectral_intensity(file_path, TL_title_name)


% DC
DC_title_name = 'DC ; 5ps FWHM ; 10modes ; modes 1-5 exited equally';
folder_DC = 'GIF625_KBSC_DC_10modes_1030nm_energy_sweep';
fiber = @DC_KBSC_10modes_1030nm;
energy = linspace(100e3, 350e3, 10);
input_energy_sweep(fiber, energy);

file_path = [folder_DC '\data_006.mat'];
plot_spectral_intensity(file_path, DC_title_name)


% UC
UC_title_name = 'UC ; 5ps FWHM ; 10modes ; modes 1-5 exited equally';
folder_UC = 'GIF625_KBSC_UC_10modes_1030nm_energy_sweep';
fiber = @UC_KBSC_10modes_1030nm;
energy = linspace(100e3, 350e3, 10);
input_energy_sweep(fiber, energy);

file_path = [folder_UC '\data_006.mat'];
plot_spectral_intensity(file_path, UC_title_name)

%%
plot_spatial_internsity(folder_DC, DC_title_name);
plot_spatial_internsity(folder_UC, UC_title_name);
plot_spatial_internsity(folder_TL, TL_title_name);


%% Functions

% plot the spatial intensity for all of the files
function plot_spatial_internsity(folder, title_name)
  
% load folder
    cd(folder);
    mfiles = ls('*.mat');
    cd('..');
    
    energy_length = size(mfiles,1);
    row_num = ceil(energy_length/5);
    
    figure;
    % title = 'DC ; 5ps FWHM ; 10modes ; modes 1-5 exited equally';
    sgtitle(title_name)
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
    
        subplot(row_num,5,i)
        imagesc(x, x, abs(time_integral).^2); colorbar;
        xlim([min(x) max(x)]);ylim([min(x) max(x)]);
        colormap(jet(128))
        axis square
        hold on
        plot(corex, corey, 'LineWidth', 1, 'Color', 'y', 'LineStyle', ':')
        hold off
        title({['Energy = ' num2str(input_field.E_tot * 1e-3) 'nJ'], ...
              ['Peak Power = ' num2str(max(sum(abs(input_field.fields).^2,2)) * 1e-3) 'kW']}); 
        % Remove axis tick labels
        set(gca, 'XTickLabel', [])
        set(gca, 'YTickLabel', [])

    end

end

% plot the spatial spectral for all of the files
function plot_spectral_intensity(file_path, title_name)

    load(file_path);

    % fix Nx issue
    if ~isfield(others, 'Nx')
        others.Nx = 1024;
    end

   
    total_field = BuildSpatialField(output_field.fields(:,:,end),fiber, sim, others);
    % time_integral = sum(abs(total_field).^2, 3) * input_field.dt;
    % [max_x_idx, max_y_idx] = find_max_indices(time_integral);

    spatial_spectal = ifft(total_field,[],3);
    spatial_spectal = fftshift(spatial_spectal , 3);

    figure;
    plot(fftshift(others.lambda), squeeze(abs( sum (sum (spatial_spectal,1) ,2) ).^2 ) );
    xlabel('Wavelength [nm]')
    ylabel('Spectral Power [a.u]')
    title({title_name, 'Integrated Spectrum'})
    xlim([960 1100])
    grid minor;
    set(gca, 'FontSize', 20);


    teta = 0:0.01:2*pi;
    corex = fiber.radius*sin(teta);
    corey = fiber.radius*cos(teta);
    dx = fiber.radius*3/others.Nx; % um
    x =  (-others.Nx/2:others.Nx/2-1)*dx;
    
    % max_idx = 700; % for TL
    % max_idx = 70; % for DC
    max_idx = 300; % for UC
    idx = round(linspace(max_idx,-max_idx,20)); % wavelengths index to plot around lambda0
    lambda0_idx = length(others.lambda)/2 + 1;
    lambda = fftshift(others.lambda);

    figure;
    sgtitle([title_name ' ; Energy= ' num2str(input_field.E_tot * 1e-3) 'nJ ; ' ...
                'Peak Power = ' num2str(max(sum(abs(input_field.fields).^2,2)) * 1e-3) 'kW']);
    
    for i = 1:length(idx)
    
        subplot(4,5,i);
        imagesc(x, x, abs(spatial_spectal(:,:,lambda0_idx+idx(i))).^2) ;
        colormap(jet(128));

        xlim([min(x) max(x)]);ylim([min(x) max(x)]);
        axis square
        hold on
        plot(corex, corey, 'LineWidth', 1, 'Color', 'y', 'LineStyle', ':')
        hold off
        title([num2str(lambda(lambda0_idx+idx(i))) 'nm']);
        % Remove axis tick labels
        set(gca, 'XTickLabel', [])
        set(gca, 'YTickLabel', [])
   
    
    end

end


function [max_x_idx, max_y_idx] = find_max_indices(coords)
    % Find the maximum value and its indices
    [~, max_idx] = max(coords(:));
    
    % Convert the linear index to row and column indices
    [max_y_idx, max_x_idx] = ind2sub(size(coords), max_idx);
end
