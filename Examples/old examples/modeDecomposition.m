[Mfield, dx] = get_modes_fields('D:\GMMNLSE\Fibers\tmpXLMA\', '1030');
%%
Mvec = zeros(size(Mfield,1).^2, size(Mfield,3));
for ii=1:size(Mfield,3)
    tmp = Mfield(:,:,ii);
    Mvec(:, ii) = tmp(:);
end
%%
x = (-size(Mfield,1)/2:size(Mfield,1)/2-1)*dx;
[xx, yy] = meshgrid(x,x);

%%
v = VideoWriter('ringoffset10.avi');
open(v);
teta = 0:0.01:2*pi;
corex = 47e-6*sin(teta);
corey = 47e-6*cos(teta);
for ii=1:55
    GW = 1e-6*ii;
%     tmp = exp(-2.77*(((xx+10*dx)/GW).^2+((yy+10*dx)/GW).^2));
%     tmp = double( sqrt((xx+10*dx).^2+(yy+10*dx).^2)<GW );
    tmp = double( sqrt((xx+10*dx).^2+(yy+10*dx).^2)<GW & sqrt((xx+10*dx).^2+(yy+10*dx).^2)>GW/2 );
%     tmp = double( sqrt((xx).^2+(yy).^2)<GW & abs(x)<5e-6);
    tmp = tmp(:);
    Mdecomp(ii,:) = tmp.'*Mvec*dx^2;
    tmpMode = Mvec*Mdecomp(ii,:).';
    tmp = reshape(tmp, 1600, 1600);
    tmpMode = reshape(tmpMode, 1600, 1600);
    Merror(ii) = sqrt(sum(abs(tmp-tmpMode).^2))/sqrt(sum(abs(tmp).^2));
%     plot(tmp(800,:))
%     hold all
%     plot(tmpMode(800,:))
    subplot(2,2,1)
    plot(Mdecomp(ii,:))
    xlabel('Mode #'); ylabel('Mode energy [a.u.]');
    subplot(2,2,2)
    plot(1:ii, Merror(1:ii))
    ylabel('Error'); ylim([0 1]);
    subplot(2,2,3)
    pcolor(x, x, abs(tmp).^2); shading flat; colormap jet;
    hold on
    plot(corex, corey, 'LineWidth', 2, 'Color', 'y')
    hold off
    axis square;
    xlim([-60e-6 60e-6]);ylim([-60e-6 60e-6]);
    subplot(2,2,4)
    pcolor(x, x, abs(tmpMode).^2); shading flat ; colormap jet; 
    hold on
    plot(corex, corey, 'LineWidth', 2, 'Color', 'y')
    hold off
    axis square;
    xlim([-60e-6 60e-6]);ylim([-60e-6 60e-6]);
    drawnow
    frame = getframe(gcf);
    writeVideo(v,frame);
    ii
end
close(v);
%%
GW = 50e-6;
tmp = exp(-2.77*(((xx+0)/GW).^2+((yy+0)/GW).^2));
plot(tmp(800,:)*2.5e4)
hold all
plot(-Mfield(800,:,1))
