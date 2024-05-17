function TfieldNew = get_modes_pinhole(Mfield, Tfield, dx, Rpinhole, offset)
Mvec = zeros(size(Mfield,1).^2, size(Mfield,3));
for ii=1:size(Mfield,3)
    tmp = Mfield(:,:,ii);
    Mvec(:, ii) = tmp(:);
end
%% create pinhole
x = (-size(Mfield,1)/2:size(Mfield,1)/2-1)*dx;
[xx, yy] = meshgrid(x,x);
pinhole = double( sqrt((xx-offset(1)).^2+(yy-offset(2)).^2)<Rpinhole );
pinhole = pinhole(:);

%% aplay SA to each point in time
TfieldNew = Tfield;
Mvec = gpuArray(Mvec);
Tfield = gpuArray(Tfield);
TfieldNew = gpuArray(TfieldNew);
pinhole = gpuArray(pinhole);
fprintf('Decomposing...   ')
for ii=1:size(Tfield,1)
    if sum(abs(Tfield(ii,:)).^2)>5e-0
        f2D = Mvec*Tfield(ii,:).';
        f2D = f2D.*pinhole;
        TfieldNew(ii,:) = (f2D.')*Mvec*dx^2;
        
    end
%     subplot(2,1,1)
%     imagesc(abs(reshape(field2D,800,800)).^2)
%     subplot(2,1,2)
%     plot(abs(Tfield(ii,:).').^2)
%     drawnow
        
end
fprintf('Done! \n')
%%
% for ii=1:size(Tfield,2)
%     figure(ii)
%     plot(abs(Tfield(:,ii)).^2)
%     hold on
%     plot(abs(TfieldNew(:,ii)).^2)
%     hold off
%     drawnow
% end

TfieldNew = gather(TfieldNew);

