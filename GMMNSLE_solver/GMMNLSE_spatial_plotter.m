% Get the mode profiles for the spatial filter
save_name = 'Nufern_incgainsat_150kWsatpower';
%save_name = 'Nufern_15modes_90nJgainsat_15kWsatpower_mmpulse';
page = 12; % 10
num_modes = 15;

Nx = 800;
mode_profiles = zeros(Nx, Nx, num_modes);
radius = '35';
lambda0 = '1030';
prefix = 'Nufern';
for ii = 1:num_modes
   name = [prefix, '/nov9_fib1_1Pradius', radius, 'boundary0000fieldscalarmode',int2str(ii),'wavelength', lambda0, '.mat'];
   load(name, 'phiplot');
   phi = phiplot;
   mode_profiles(:, :, ii) = phi;
   disp(['Loaded mode ', int2str(ii)])
end
load(name, 'xf');
x = xf-mean(xf);

norms = zeros(num_modes, 1);
for midx = 1:num_modes
    norms(midx) = sqrt(sum(sum(abs(mode_profiles(:, :, midx)).^2)));
end

startpoint = 448;
endpoint = 448;

spatial_max_field = zeros(Nx, Nx, endpoint-startpoint+1);

for rt_num = startpoint:endpoint
    load([save_name '_' num2str(rt_num) '.mat']);
    
    N = size(output.fields, 1);
    dt = output.dt;
    output_field = output.fields(:, :, page);
    [maxtime, argmaxtime] = max(output_field); % max of each column
    [maxmode, argmaxmode] = max(maxtime); % max mode
    time_point = argmaxtime(argmaxmode);
    disp((time_point-N/2)*dt);
    disp(sum(abs(output_field(:)).^2)*output.dt/10^3);
    
    for midx = 1:num_modes
        spatial_max_field(:, :, rt_num - startpoint+1) = spatial_max_field(:, :, rt_num - startpoint+1) + ...
            mode_profiles(:, :, midx)*output_field(time_point, midx)/norms(midx);
    end
end

%%

[X, Y] = meshgrid(x, x);

figure('Position', [200, 50, 900, 800]);


for rt_num = startpoint:endpoint
%for rt_num = 38:44
    h = pcolor(X, Y, abs(spatial_max_field(:, :, rt_num-startpoint+1)).^2);
    h.LineStyle = 'none';
    colorbar;
    axis square;
    xlabel('x (um)');
    ylabel('y (um)');
    xlim([-200, 200]);
    ylim([-200, 200]);
    pause(1);
end

