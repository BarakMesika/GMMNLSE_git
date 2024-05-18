% by BarakM 14.5.24
% INPUT:
% fields - (t,mode) the field of each mode in time (for a given z value)
% fiber - fiber parameters (spesificly the fiber folder)

% OUTPUT:
% total_field - (x,y,t) function of the spatial location and time


function total_field = BuildSpatialField(fields,fiber, sim, others)

t = others.t;
lambda = sim.lambda0;

% downsample the spatial picture by a factor of n^2
n = 4;
 if mod(others.Nx, n) ~= 0
        error('Matrix size must be divisible by n');
 end
 newSize = others.Nx / n;

% TODO: add all z
total_field = zeros( newSize, newSize, length(others.t) ); % total_field(X,Y,t)

h = waitbar(0, 'calculate field...');

% TODO: try using parfor loop
    for ii=1:others.modes

        % load mode spashial profile
        fname=[fiber.MM_folder 'radius' strrep(num2str(fiber.radius), '.', '_') 'boundary0000fieldscalar'...
                'mode' num2str(ii,'%03.f') 'wavelength' strrep(num2str(lambda*1e9), '.', '_')];
        phi = load([ fname '.mat'], 'phi');
        phi = phi.phi;
		
		phi_downsampled = downsample_matrix(phi, n);

        % normalize the field norm
        norm_mode = (phi_downsampled) ./ sqrt((sum(sum(abs(phi_downsampled).^2))));

        % ploting the mode and the downsample
        % imagesc(abs(phi).^2);
        % imagesc(abs(phi_downsampled).^2);
        % imagesc(abs(norm_mode).^2);
        % sum(sum(abs(norm_mode).^2))


        tmp = zeros(newSize, newSize, length(others.t));
        for idx_t = 1:length(t)
            tmp(:,:,idx_t) = norm_mode(:,:) .* fields(idx_t,ii) ;
        end
        total_field = total_field + tmp;
        % dt = others.t(2)-others.t(1);
        % sum(sum(sum(abs(total_field).^2))) * dt; % make sure that the integral is the pulse energy

        waitbar(ii/others.modes, h, ['mode ' num2str(ii) ' from ' num2str(others.modes)]);


    end
    close(h);

end



% TODO: make avreging block
function downsampledMatrix = downsample_matrix(originalMatrix, n)


    N = size(originalMatrix,1);
    newSize = N/n;

    downsampledMatrix = zeros(newSize, newSize);

    for i = n:n:N
        for j = n:n:N
            downsampledMatrix(i/n,j/n) = originalMatrix(i,j);
        end
    end

    % % Reshape the matrix into nxn blocks
    % reshaped = reshape(originalMatrix, n, n, newSize, newSize);
    % 
    % % Calculate the average of each nxn block
    % downsampledMatrix = squeeze(mean(reshaped, [1 2]));

end


