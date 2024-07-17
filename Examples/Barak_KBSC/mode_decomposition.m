%% load the 2D matrix of the spatial energy
full = sqrt(sum(SSdata,3)); % sqrt to get the aproximate field (without phase)

% % zero values that uder a sertain threshold
% 
% % Create a logical mask for values below the threshold
% mask = full < 2.1e-5;
% % Zero out the values below the threshold
% zeroedMatrix = full;
% zeroedMatrix(mask) = 0;
% % imagesc(zeroedMatrix);

%% calculate the coeffitiants for each mode
% run from the fiber modes folder

load('Fiber_params.mat')
x_size = size(full,1);
y_size = size(full,2);
reconstrucPhi = zeros( x_size, y_size );

for ii=1:data.num_modes

        % load mode spashial profile
        fname=['radius' strrep(num2str(data.radius), '.', '_') 'boundary0000fieldscalar'...
                'mode' num2str(ii,'%03.f') 'wavelength' strrep(num2str(data.lambda0*1e9), '.', '_')];
        phi = load([ fname '.mat'], 'phi');
        phi = phi.phi;
        
        downsampledPhi = imresize( phi, [x_size, y_size] );
        downsampledPhi_norm = downsampledPhi ./ sqrt((sum(sum(abs(downsampledPhi).^2))));
        
        c(ii) = sum( sum (full .* (downsampledPhi) ) );

        reconstrucPhi = (c(ii) .* downsampledPhi) + reconstrucPhi;
        
end


%% plot original and reconstructed
figure;

% subplot(1,2,1); imagesc(abs(reconstrucPhi).^2); title('Reconstructed');
subplot(1,2,1); imagesc(time_integral); title('Reconstructed');

subplot(1,2,2); imagesc(full); title('Original');

%% normalize vlaues
% modes_coeff = sqrt(c);
% modes_coeff = c ./ sum(abs(c));
% stem(abs(modes_coeff));

%% save vlaues
save('modes_coef', 'c');
close;
