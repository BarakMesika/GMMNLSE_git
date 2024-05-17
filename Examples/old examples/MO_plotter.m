%% MO plotter
dirName = 'D:\GMMNLSE\Examples\GRIN30Yb\Filter2_3\';
fname = ls([dirName 'data1*.*']);
load([dirName 'simParam.mat']);
dt = data.t(2)-data.t(1);
%%
opengl hardware;
set(0, 'DefaultFigureRenderer', 'opengl');
set(0, 'defaultAxesFontSize', 16);
figure('pos', [100 100 600 600]);
for ii=1:size(fname,1)
    load([dirName 'data1_RT' num2str(ii) '.mat']);
    subplot(3,2,1)
    plot(data.t, abs(uamp).^2, 'LineWidth', 2)
    xlim([-15 15]); xlabel('Time [ps]');
    subplot(3,2,2)
    plot(fftshift(data.lambda), abs(fftshift(ifft(uamp, [], 1),1)).^2, 'LineWidth', 2)
    xlim([980 1200]); xlabel('\lambda [nm]');
    E1(ii,:) = sum(abs(uamp).^2)*dt*1e-3;
    
    
    load([dirName 'data2_RT' num2str(ii) '.mat']);
    subplot(3,2,3)
    plot(data.t, abs(uamp).^2, 'LineWidth', 2)
    xlim([-15 15]); xlabel('Time [ps]');
    subplot(3,2,4)
    plot(fftshift(data.lambda), abs(fftshift(ifft(uamp, [], 1),1)).^2, 'LineWidth', 2)
    xlim([980 1200]); xlabel('\lambda [nm]');
    E2(ii,:) = sum(abs(uamp).^2)*dt*1e-3;
    
    subplot(3,2,5:6)
    plot(1:ii, E1(1:ii,:)+E2(1:ii,:), 'LineWidth', 2);
    xlabel('Round trip#'); ylabel('Pulse energy [nJ]')
    legend({'Mode1', 'Mode2', 'Mode3', 'Mode4', 'Mode5'})
    pause(0.2)
end


