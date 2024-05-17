fDir = 'D:\GMMNLSE\SimData\Oscillator_PLMA-YDF-30-250-VIII\sat250nJFilter06nm\';
fname = ls([fDir 'data_*.mat']);
load('D:\GMMNLSE\SimData\Oscillator_PLMA-YDF-30-250-VIII\SimParam.mat');
%%
v = VideoWriter('oscillator_simple_4comb.avi');
open(v);
cmap = hsv(size(SimParam.Modes,3));
figure();
set(gcf, 'Position', [200 200 800 600]);
for ij=1:size(fname,1)
   load([fDir fname(ij,:)])
   
   subplot(2,2,1)
   for ii=1:size(input_field.fields,2)
        plot(SimParam.time, abs( uout(:,ii) ).^2, 'LineWidth', 2, 'Color', cmap(ii,:));
        hold on
        xlabel('Time [ps]');ylabel('Power [W]');
        
   end
   xlabel('Time [ps]');ylabel('Power [W]');
   xlim([min(SimParam.time) max(SimParam.time)]);ylim([0 15000]);
   drawnow
   hold off
    
   subplot(2,2,2)
   for ii=1:size(input_field.fields,2)
        plot(SimParam.wl, abs( (ifft(uout(:,ii))) ).^2, 'LineWidth', 2,  'Color', cmap(ii,:));
        hold on
        xlabel('Time [ps]');ylabel('Power [W]');
        
   end
   xlabel('\lambda [nm]');ylabel('Spectrum [a.u.]');
   xlim([980 1120]);ylim([0 1.2]);
   drawnow
   hold off
   
   E(:,ij) = sum(abs(uout).^2,1)*SimParam.dt/1e3;
   subplot(2,2,3)
   for ii=1:size(input_field.fields,2)
       plot(1:ij, E(ii,1:ij), 'LineWidth', 2,  'Color', cmap(ii,:))
       hold on
   end
   hold off
   ylabel('Energy [nJ]');
   xlabel('Round trip');
   xlim([0 ij])
   
   Ixy = buildIxy(uout, SimParam);
   subplot(2,2,4)
   him = imagesc(Ixy); colormap jet;
   axis square
   set(gca, 'XTick', [])
   set(gca, 'YTick', [])
   ij
   frame = getframe(gcf);
   writeVideo(v,frame);
end
close(v);
%%
figure('pos', [200 200 400 300]);
for ii=1:size(input_field.fields,2)
    subplot(3,3,ii)
    imagesc(abs(SimParam.Modes(:,:,ii)).^2)
    colormap jet;
    set(gca, 'XTick', []);
    set(gca, 'YTick', []);
    axis square
    xlim([400-200 400+200]);
    ylim([400-200 400+200]);
end