% Load modes and parameters
num_modes = 10;
polynomial_fit_order = 8;
num_disp_orders = 4;

for ik=1:num_modes    
    fname = ls(['LP_000-0' num2str(ik,'%02.f') '*.*']);
    for ii=1:size(fname,1)
       load(fname(ii,:), 'neff', 'wavelength'); 
       n(ik, ii) = neff;
       lambda(ii) = wavelength;
       clear neff;
    end
end
% approximate the dispersion parameters
c = 2.99792458e-4; % speed of ligth m/ps
lambda0 = 1030e-9;
f0 = c/lambda0; % center frequency in THz
f = c./lambda;
w=2*pi*f;
w_disp = 2*pi*c/lambda0;
beta_calc = n'.*w'/c;


b_coefficients = zeros(num_modes, num_disp_orders+1); % The dispersion coefficients
for midx = 1:num_modes
    beta_calc_i = beta_calc(:, midx);
    
    beta_fit_last = polyfit(w', beta_calc_i, polynomial_fit_order); % the fit coefficients
    b_coefficients(midx, 1) = polyval(beta_fit_last, w_disp)/1000; % Save beta_0 in 1/mm
    for disp_order = 1:num_disp_orders
        % The derivatives can be calculated exactly from the coefficients
        beta_fit_last = ((polynomial_fit_order-(disp_order-1)):-1:1).*beta_fit_last(1:(polynomial_fit_order-(disp_order-1)));
        b_coefficients(midx, disp_order+1) = polyval(beta_fit_last, w_disp)*(10^3)^disp_order/1000; % beta_n in fs^n/mm
    end
end

% beta0 and beta1 should be relative to the fundamental mode.
b_coefficients(:, 1) = b_coefficients(:, 1) - ones(num_modes, 1)*b_coefficients(1, 1);
b_coefficients(:, 2) = b_coefficients(:, 2) - ones(num_modes, 1)*b_coefficients(1, 2);

bettas = b_coefficients;
%%
SubPlotNum = ceil( sqrt(num_disp_orders) );
for ii=1:num_disp_orders
    subplot(SubPlotNum,SubPlotNum,ii)
    plot(1:num_modes, bettas(:,ii),'ob');
    grid on
    xlabel('LP_0_m mode')
    ylabel(['\beta_' num2str(ii-1) 'fs^' num2str(ii-1) '/mm'])
end


