function output = redefine_field_time_window( input,initial_central_wavelength,...
                                                     final_central_wavelength,final_dt,final_Nt)
%REDEFINE_FIELD_TIME_WINDOW It re-computes the fields based on the time window.

initial_Nt = size(input.fields,1);
initial_dt = input.dt;

initial_time_window = initial_Nt*initial_dt;
final_time_window = final_Nt*final_dt;

abs_fields = abs(input.fields);
phase_fields = unwrap(angle(input.fields));

initial_t = (-floor(initial_Nt/2):ceil(initial_Nt/2-1))'*(initial_time_window/initial_Nt);
final_t = (-floor(final_Nt/2):ceil(final_Nt/2-1))'*(final_time_window/final_Nt);

abs_fields = interp1(initial_t,abs_fields,final_t,'pchip',0);
phase_fields = interp1(initial_t,phase_fields,final_t,'pchip',0);

c = 299792458*1e-12; % m/ps
output_field = abs_fields.*exp(1i*(phase_fields+2*pi*(c./final_central_wavelength-c./initial_central_wavelength).*final_t));

output = input;
output.fields = output_field;
output.dt = final_dt;

end