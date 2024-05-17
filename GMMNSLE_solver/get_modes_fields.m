function [field, dx] = get_modes_fields(FolderName, lambda)
fname = ls(fullfile(FolderName,'radius*.mat'));
ind = 1;
for ii=1:size(fname,1)
    if contains(fname(ii,:), ['wavelength' lambda])
        load(fullfile(FolderName, fname(ii,:)), 'phi');
        load(fullfile(FolderName, fname(ii,:)), 'x');
        dx = (x(2)-x(1))*10^-6; % spatial step in m
        field(:,:,ind) = phi/sqrt(dx^2*sum(sum(abs(phi).^2)));
        ind = ind+1;
    end
end




