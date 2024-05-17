function vn = calc_vn(vi,vo, mode)
%CALC_VN Summary of this function goes here
%   Detailed explanation goes here

v = vo./vi;

switch mode
    case 'TE'
        vn = 1./sqrt(v.^2-1);
    case 'TM'
        vn = v.^2./sqrt(v.^2-1);
    case 'EH'
        vn = 1/2*(v.^2+1)./sqrt(v.^2-1);
end

end

