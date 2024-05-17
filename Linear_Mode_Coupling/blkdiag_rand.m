function func = blkdiag_rand(sim)
%BLKDIAG_RAND It generates block-diagonal random matrices, each with 
%a certain type of special matrix. It can generate unitary,
%close-to-identity unitary, hermitian, and skew-hermitian random matrices.
%
%   func.haar - totally unitary random matrix
%   func.identity_exp - close-to-identity unitary random matrix
%   func.identity_rootn - close-to-identity unitary random matrix
%   func.hermitian - hermitian random matrix
%   func.skew_hermitian - skew-hermitian random matrix
%
%   func.single_yes - whether to use "single precision" or not
%   func.gpu_yes - whether to use "gpu" or not
%
%   For more information, please look into
%   "random_unitary_matrix_generator.m" or
%   "random_special_matrix_generator.m"

if exist('sim','var')
    % Unitary matrix
    fun1 = random_unitary_matrix_generator(sim);
    % Hermitian, Skew-Hermitian
    fun2 = random_special_matrix_generator(sim);
    
    % status: whether to use single, gpu or not
    status = struct('single_yes',sim.single_yes,'gpu_yes',sim.gpu_yes);
else
    % Unitary matrix
    fun1 = random_unitary_matrix_generator;
    % Hermitian, Skew-Hermitian
    fun2 = random_special_matrix_generator;
    
    % status: whether to use single, gpu or not
    status = struct('single_yes',false,'gpu_yes',false);
end

func = catstruct(fun1,fun2,status);

end

