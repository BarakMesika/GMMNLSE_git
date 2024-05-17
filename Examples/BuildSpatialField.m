% by BarakM 14.5.24
% INPUT:
% output_field - (t,z,mode) the field of each mode in time
% fiber - fiber parameters (spesificly the fiber folder)

% OUTPUT:
% total_field - (x,y,z,t) function of the location and time

% TODO: - solve memory problem (needs 128GB ram now...)
%       - add also for each z . now it is only at the end of the fiber

function total_field = BuildSpatialField(output_field,fiber, sim, others)

z =  0:sim.save_period:fiber.L0;
t = others.t;
lambda = sim.lambda0;
x = (-others.Nx/2) : 1 : others.Nx/2 - 1;

% TODO: add all z
total_field = zeros( others.Nx, others.Nx, 1, length(others.t) ); % total_field(X,Y,Z,t)

    for ii=1:others.modes

        % load mode spashial profile
        fname=[folder_name sep_char 'radius' strrep(num2str(fiber.radius), '.', '_') 'boundary0000field' field ...
                'mode' num2str(ii,'%03.f') 'wavelength' strrep(num2str(lambda*1000), '.', '_')];
        load([fname '.mat'], 'phi');

        % normalize the field norm
        norm_mode = phi ./ sqrt(sum(sum(abs(phi).^2)));


        % only at the end of the fiber
        % TODO: for all Z
        for idx_t = 1:length(t)
            total_field(:,:,t,end) = total_field(:,:,t,end) + norm_mode*output_field.fields(idx_t,ii,end);
        end

    end
    

end



