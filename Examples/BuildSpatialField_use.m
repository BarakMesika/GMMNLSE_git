%% try to build the energy for each mode
% t = others.t;
% dt = input_field.dt;
% T0 = 60e-3; % 70fs
% tmp = exp(-1*(t/T0).^2);                    % init pulse shape (will be notmalized to 1nJ)
% 
% E_modes = zeros(1,others.modes);
% input_field.E_tot = 40e3;                                      % Total Energy [pJ]
% E_modes(2:3) = 1/4; 
% E_modes(5:8) = 1/4; 
% 
% % normlized energy to 1pJ
% tmp = tmp/sqrt( dt*sum(abs(tmp).^2));
% 
% E_modes = E_modes * input_field.E_tot;
% 
% % give energy to the initial pulse. for each mode
% for ii=1:others.modes
%     input_field.fields(:,ii) = sqrt(E_modes(ii))*tmp ;
% end
% 
% total_field = BuildSpatialField(input_field.fields,fiber, sim, others);

%% with real simulation
total_field = BuildSpatialField(output_field.fields(:,:,end),fiber, sim, others);

% make integral on all time ~20ps
time_integral = sum(abs(total_field).^2, 3) * input_field.dt;
% energy_tot = sum(sum(time_integral)) % check that we get all of the energy
%%
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
        

