clearvars; %close all;

num_disp_orders = 2;

load('info_25um.mat','beta','wavelength');

c = 2.99792458e-4; % speed of ligth m/ps

Nf = size(beta,1);
%num_modes = size(beta,2);
num_modes = 1;

lmin = 0.2;
lmax = 2;

%% Calculate the propagation constants
l = wavelength*1e6; % um
fo = c./l*1e6; % THz

f = linspace(fo(end),fo(1),Nf)';
abs_beta = interp1(fo,abs(beta),f,'pchip');
ang_beta = interp1(fo,unwrap(angle(beta),[],1),f,'pchip');
beta = abs_beta.*exp(1i*ang_beta);

w = 2*pi*f; % angular frequencies in 1/ps
df = f(2)-f(1);
dw = 2*pi*df;
beta_calc = real(beta); % beta in 1/m

%% Display the results

% We need to use cell arrays because the higher orders are calculated from
% finite differences. This means that each order has one less data point
% than the previous one.
w_vectors = cell(num_disp_orders+1, 1); % omegas, in 1/ps
l_vectors = cell(num_disp_orders+1, 1); % lambdas, in um
w_vectors{1} = w;
l_vectors{1} = 2*pi*c./w_vectors{1}*1e6;
for disp_order = 1:num_disp_orders
    w_prev = w_vectors{disp_order};
    w_vectors{disp_order+1} = dw/2 + w_prev(1:length(w_prev)-1); % in 1/ps
    l_vectors{disp_order+1} = 2*pi*c./w_vectors{disp_order+1}*1e6; % in um
end

% beta_full will have all of the orders, for each mode, as a function of
% wavlength
beta_full = cell(num_disp_orders+1, 1);
beta_full{1} = beta_calc/1000;
for disp_order = 1:num_disp_orders
    beta_full{disp_order+1} = zeros(Nf-disp_order, num_modes);
end

% Take the differences to calculate the higher orders
for midx = 1:num_modes
    for disp_order = 1:num_disp_orders
        beta_full{disp_order+1}(:, midx) = diff(beta_full{disp_order}(:, midx))/dw*1000;
    end
end

coo=hsv(num_modes);

ylabels = cell(num_disp_orders+1, 1);
ylabels{1} = '1/mm';
ylabels{2} = 'fs/mm';
for disp_order = 2:num_disp_orders
    ylabels{disp_order+1} = ['fs^' num2str(disp_order) '/mm'];
end

% Plot the results
figure;
for disp_order = 1:num_disp_orders+1
    subplot(1,num_disp_orders+1,disp_order)
    hold on
    for midx = 1:num_modes
        h = plot(l_vectors{disp_order}, beta_full{disp_order}(:, midx), 'Color', coo(midx,:));
        set(h,'linewidth',2);
        xlim([lmin,lmax]);
    end
    hold off
    set(gca,'fontsize',20);
    ylabel(ylabels{disp_order});
    xlabel('\mum')
    title(['\beta_' num2str(disp_order-1)]);
end