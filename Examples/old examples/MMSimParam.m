classdef MMSimParam
    %Results from GMMNLSE
    properties (SetAccess = public)
        pulses
        Modes
        time
        f
        wl
        wl0
        dt
        c
        dx
        N
    end
    
    methods
        function obj = MMSimParam()
            %MMSimRes Construct an instance of this class     
        end
        function Ixy = buildIxy(pulse, SimParam)
            Mvec = zeros(size(SimParam.Modes,1).^2, size(SimParam.Modes,3));
            for ii=1:size(SimParam.Modes,3)
                tmp = SimParam.Modes(:,:,ii);
                Mvec(:, ii) = tmp(:);
            end
            spec = fftshift(fft(pulse,[],1),1);
            ind = find( sum( abs(spec).^2, 2)>500 );
            Ixy = zeros(size(SimParam.Modes,1).^2,1);
            Ixy = gpuArray(Ixy);
            Mvec = gpuArray(Mvec);
            pulse = gpuArray(pulse);
            spec = gpuArray(spec);
            
            fprintf('Decomposing...   ')
            parfor ii=1:size(ind,1)
                    Ixy = Ixy + abs(Mvec*spec(ind(ii),:).').^2;   
            end
            fprintf('Done! \n')
            Ixy = reshape(Ixy, size(SimParam.Modes,1), size(SimParam.Modes,1));
        end
    end
end

