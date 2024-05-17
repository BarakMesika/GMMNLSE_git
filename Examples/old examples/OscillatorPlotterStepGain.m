fDir = 'D:\GMMNLSE\SimData\Oscillator_HighPower_PLMA-YDF-30-250-VIII\PH016um_sat100\';
fname = ls([fDir 'data_*.mat']);
load('D:\GMMNLSE\SimData\Oscillator_HighPower_PLMA-YDF-30-250-VIII\simParam.mat');
%%
v = VideoWriter('oscillator_PLMA-YDF-30-250-VIII_PH016um.avi');
open(v);
cmap = lines(8);
figure();
set(gcf, 'Position', [200 200 800 500]);
for ij=1:size(fname,1)
   load([fDir fname(ij,:)])
   
   
   for ii=1:size(output_field.fields,3)
        subplot(1,2,1)
        plot(SimParam.time, abs( output_field.fields(:,:,ii) ).^2, 'LineWidth', 2);
        xlabel('Time [ps]');ylabel('Power [W]');
        xlim([min(SimParam.time) max(SimParam.time)]);ylim([0 30000]);
        
        subplot(1,2,2)
        spec = ifft(output_field.fields(:,:,ii),[],1);
        plot(SimParam.wl, abs( spec ).^2, 'LineWidth', 2);
        xlabel('\lambda [nm]');ylabel('Spectrum [a.u.]');
        xlim([980 1120]);ylim([0 15]);
        drawnow
        
        for mi=1:size(output_field.fields,2)
            E(mi) = sum(abs(output_field.fields(:,mi,ii)).^2,1)*SimParam.dt/1e3;
        end
        E = fix(E*10)/10;
        annstr = ['E_1=' num2str(E(1)) ' E_2=' num2str(E(2)) ' E_3=' num2str(E(3))...
            ' E_4=' num2str(E(4)) ' E_5=' num2str(E(5)) ' E_6=' num2str(E(6))...
            ' E_7=' num2str(E(7)) ' E_8=' num2str(E(8))];
        annstr = strcat(['RT:' num2str(ij) '  ---  ' 'Z:' num2str(output_field.z(ii)) '  ---  '], annstr);
        annh = annotation(gcf,'textbox', [0.11 0.95 0.85 0.065],...
            'String',{annstr},'LineStyle','none', 'FitBoxToText','on', 'FontSize',14);
        
        frame = getframe(gcf);
        writeVideo(v,frame);
        delete(annh);
   end
   
%    E(:,ij) = sum(abs(uout).^2,1)*SimParam.dt/1e3;

end
close(v);