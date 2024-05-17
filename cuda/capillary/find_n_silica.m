clearvars; close all;

addpath('SLMtools');

% refractive index
data = dlmread('n_silica (Malitson).csv',',',1,0);
data_wl = data(:,1); % um
data_n = data(:,2);

%% Sellmeier formula
wavenumber_R = 1./data_wl;

a = [0.6961663,0.4079426,0.8974794];
b = [0.0684043,0.1162414,9.896161];
Sellmeier_terms = @(lambda,a,b) a.*lambda.^2./(lambda.^2 - b.^2);
refractivity = @(lambda) sqrt(1+sum(Sellmeier_terms(lambda,a,b),2));
n_Sellmeier = refractivity(1./wavenumber_R);

%% Extended range of wavelengths
wavelength_L = [linspace(0.117,0.2,100),linspace(0.21,7,100)]';
wavelength_R = linspace(6.75,15,30)'; % um
wavenumber_L = 1./wavelength_L;
wavenumber_R = 1./wavelength_R;

n_Sellmeier_calc_L = refractivity(1./wavenumber_L);

slm = slmengine(wavelength_L,n_Sellmeier_calc_L,'knots',wavelength_L,... 
   'decreasing','on','plot','on','extrapolation','constant');

n_data_calc = slmeval(wavelength_R,slm,0); % refractive index of H2

%% Plot
figure;
h = plot(data_wl,[data_n,n_Sellmeier]);
legend('Data','Fitted Sellmeier');
set(h,'linewidth',2);
xlabel('Wavelength (\mum)'); ylabel('n');
title('Loaded data');

figure;
h = plot([wavelength_L;wavelength_R],[n_Sellmeier_calc_L;n_data_calc]);
legend('Fitted Data');
set(h,'linewidth',2);
xlabel('Wavelength (\mum)'); ylabel('n');
title('Extended wavelength');

%% Save
save('n_silica.mat','slm');