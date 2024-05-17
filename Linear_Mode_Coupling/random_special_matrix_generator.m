function fun = random_special_matrix_generator(sim)
%RANDOM_SPECIAL_MATRIX_GENERATOR It generates block-diagonal Hermitian and skew-Hermtian matrices
%   Usage:
%       (1) First generate the function handle,
%
%           fun = random_special_matrix_generator;
%           fun = random_special_matrix_generator(sim);
%           
%           "sim" has sim.gpu_yes - 0 or 1 (true or false); use gpuArray or not
%                     sim.single_yes - 0 or 1 (true or false);
%                                      use the type "single" or "double" for matrices
%           If no "sim", it'll run with CPU and "double precision".
%
%       (2) Then call the function and generate the output.
%           
%           random_matrix = fun.hermitian(n,n3);
%           random_matrix = fun.skew_hermitian(n,n3);
%           
%           n - an array specifying the dimension (n,n) of the block-diagonal random matrices,
%               where each block is a unitary matrix
%           n3 - the number of random matrices to generate
%           random_matrix (output) - (n,n,n3) matrix with each nxn a random matrix
%

if exist('sim','var')
    fun.hermitian = @(n,n3) rand_hermitian(n,n3,sim);
    fun.skew_hermitian = @(n,n3) rand_skew_hermitian(n,n3,sim);
else
    fun.hermitian = @(n,n3) rand_hermitian(n,n3);
    fun.skew_hermitian = @(n,n3) rand_skew_hermitian(n,n3);
end

end

%% RAND_HERMITIAN
function rand_matrix = rand_hermitian(n,n3,sim)
%RAND_HERMITIAN It generates a random hermitian matrix.

% "n" should be in a row, not a column.
if size(n,1) > 1
    n = n';
end

if n == 1
    if exist('sim','var')
        rand_matrix = matrix_initialization(n,n3,sim,@randn);
    else
        rand_matrix = matrix_initialization(n,n3,@randn);
    end
else
    
    % Initialization
    if exist('sim','var')
        rand_matrix = matrix_initialization(n,n3,sim,@zeros);
    else
        rand_matrix = matrix_initialization(n,n3,@zeros);
    end
    
    sn = sum(n);
    
    % Point out the linear indices of independent varibles within upper-half
    % Hermitian.
    previous_n = cumsum(n);
    previous_n = [0 previous_n(1:end-1)];
    num_independent_elements = [0 (n-1).*n/2];
    previous_num_independent_elements = cumsum(num_independent_elements);
    idx = zeros(1,sum(num_independent_elements),'uint32');
    for i = 1:length(n)
        nidx = n(i);
        for m = 1:(nidx-1)
            idx(previous_num_independent_elements(i) + (m-1)*m/2 + (1:m)) = (previous_n(i)+m)*sn + previous_n(i) + (1:m);
        end
    end
    n3idx_increment = uint32(permute(((1:n3)-1)*sn^2,[1 3 2]));
    idx = bsxfun(@plus, idx,n3idx_increment);
    
    % Put random numbers in these slots of independent variables.
    rand_matrix(idx(:)) = (randn(1,sum(num_independent_elements)*n3) + 1i*randn(1,sum(num_independent_elements)*n3))/sqrt(2);
    
    % Calculate the lower-half part according to the upper-half part.
    if exist('sim','var') && sim.gpu_yes
        rand_matrix = rand_matrix + pagefun(@ctranspose, rand_matrix);
    else
        rand_matrix = rand_matrix + conj(permute(rand_matrix,[2 1 3]));
    end
    
    % Diagonal part: real numbers
    idx = bsxfun(@plus, uint32( (1:sn) + ((1:sn)-1)*sn ), n3idx_increment);
    rand_matrix(idx(:)) = randn(1,sn*n3);
end

end

%% RAND_SKEW_HERMITIAN
function rand_matrix = rand_skew_hermitian(n,n3,sim)
%RAND_HERMITIAN It generates a random skew-hermitian matrix.

% "n" should be in a row, not a column.
if size(n,1) > 1
    n = n';
end

if n == 1
    if exist('sim','var')
        rand_matrix = 1i*matrix_initialization(n,n3,sim,@randn);
    else
        rand_matrix = 1i*matrix_initialization(n,n3,@randn);
    end
else
    
    % Initialization
    if exist('sim','var')
        rand_matrix = matrix_initialization(n,n3,sim,@zeros);
    else
        rand_matrix = matrix_initialization(n,n3,@zeros);
    end
    
    sn = sum(n);
    
    % Point out the linear indices of independent varibles within upper-half
    % Hermitian.
    previous_n = cumsum(n);
    previous_n = [0 previous_n(1:end-1)];
    num_independent_elements = [0 (n-1).*n/2];
    previous_num_independent_elements = cumsum(num_independent_elements);
    idx = zeros(1,sum(num_independent_elements),'uint32');
    for i = 1:length(n)
        nidx = n(i);
        for m = 1:(nidx-1)
            idx(previous_num_independent_elements(i) + (m-1)*m/2 + (1:m)) = (previous_n(i)+m)*sn + previous_n(i) + (1:m);
        end
    end
    n3idx_increment = uint32(permute(((1:n3)-1)*sum(n)^2,[1 3 2]));
    idx = bsxfun(@plus, idx,n3idx_increment);
    
    % Put random numbers in these slots of independent variables.
    rand_matrix(idx(:)) = (randn(1,sum(num_independent_elements)*n3) + 1i*randn(1,sum(num_independent_elements)*n3))/sqrt(2);
    
    % Calculate the lower-half part according to the upper-half part.
    if exist('sim','var') && sim.gpu_yes
        rand_matrix = rand_matrix - pagefun(@ctranspose, rand_matrix);
    else
        rand_matrix = rand_matrix - conj(permute(rand_matrix,[2 1 3]));
    end
    
    % Diagonal part: imaginary numbers
    idx = bsxfun(@plus, uint32( (1:sn) + ((1:sn)-1)*sn ), n3idx_increment);
    rand_matrix(idx(:)) = 1i*randn(1,sn*n3);
end

end

%% MATRIX_INITIALIZATION
function varargout = matrix_initialization(n,n3,varargin)
%MATRIX_INITIALIZATION It initializes (sum(n),sum(n),n3) matrix according to "sim" and
%the functions defined in varargin. It also works with block-diagonal
%initialization for "randn".

if isstruct(varargin{1})
    sim = varargin{1};
    varargin = varargin(2:end);
else
    sim = struct('gpu_yes',false,'single_yes',false);
end

nfun = length(varargin);

varargout = cell(1,nfun);
single_matrix = cell(1,length(n));
for fidx = 1:nfun
    switch func2str(varargin{fidx})
        case 'randn'
            for nidx = 1:length(n)
                if sim.gpu_yes
                    if sim.single_yes
                        single_matrix{nidx} = feval(varargin{fidx},n(nidx),n(nidx),n3,'single','gpuArray');
                    else
                        single_matrix{nidx} = feval(varargin{fidx},n(nidx),n(nidx),n3,'gpuArray');
                    end
                else
                    if sim.single_yes
                        single_matrix{nidx} = feval(varargin{fidx},n(nidx),n(nidx),n3,'single');
                    else
                        single_matrix{nidx} = feval(varargin{fidx},n(nidx),n(nidx),n3);
                    end
                end
            end

            if length(n) == 1
                varargout{fidx} = single_matrix{1};
            else
                varargout{fidx} = matrix_initialization(sum(n),1,sim,@zeros);
                for n3idx = 1:n3
                    single_matrix_in_n3idx = cellfun(@(m) m(:,:,n3idx), single_matrix,'UniformOutput',false);
                    varargout{fidx}(:,:,n3idx) = blkdiag(single_matrix_in_n3idx{:});
                end
            end
        case {'zeros','ones'}
            if sim.gpu_yes
                if sim.single_yes
                    varargout{fidx} = feval(varargin{fidx},sum(n),sum(n),n3,'single','gpuArray');
                else
                    varargout{fidx} = feval(varargin{fidx},sum(n),sum(n),n3,'gpuArray');
                end
            else
                if sim.single_yes
                    varargout{fidx} = feval(varargin{fidx},sum(n),sum(n),n3,'single');
                else
                    varargout{fidx} = feval(varargin{fidx},sum(n),sum(n),n3);
                end
            end
        otherwise
            error('Not supported yet.');
    end
end

end