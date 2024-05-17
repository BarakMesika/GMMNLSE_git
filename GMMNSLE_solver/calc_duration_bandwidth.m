function [ duration,bandwidth ] = calc_duration_bandwidth( t,wavelength,field )
%CALC_DURATION_BANDWIDTH It calculates the RMS duration and bandwidth.
%
%   t: (N,1); time
%   wavelength: (N,1); wavelength
%   field: (N,......), a multidimensional array composed of columns of
%          fields to be calculated

duration = calc_RMS(t,abs(field).^2);
bandwidth = calc_RMS(wavelength,abs(field).^2);

end