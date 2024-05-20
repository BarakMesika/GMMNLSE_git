data1 = 'data_011';
data2 = 'data_009';

compare_data(data1, data2, 1);



% INPUT: data1,data2 - names of the file that store the output data
%        disp_mode - for which mode to calculate
function compare_data(data1, data2, disp_mode)
out1 = load(data1, 'output_field');
out2 = load(data2, 'output_field');

load(data1);
dt = abs( others.t(2) - others.t(1) );
dwl = abs( others.lambda(2) - others.lambda(1) );
for jj=1:(fiber.L0/sim.save_period)

    ii=disp_mode;
    Fout1 = fftshift(ifft(out1.output_field.fields(:,ii,jj)),1);
    Fout2 = fftshift(ifft(out2.output_field.fields(:,ii,jj)),1);

    time_dif_nurm(jj) = norm( out1.output_field.fields(:,ii,jj) - out2.output_field.fields(:,ii,jj) ) * dt /...
                        sqrt( norm(out1.output_field.fields(:,ii,jj)) * norm(out2.output_field.fields(:,ii,jj)) );
    spectrum_dif_norm(jj) = norm( Fout1 - Fout2 ) * dwl / sqrt( norm(Fout1) * norm(Fout2) );
    
    
end

% display for mode

x = 0:sim.save_period:fiber.L0;
x = x(1:end-1);
figure;
sgtitle(['Diff for mode ' num2str(disp_mode)] )

subplot(2,1,1);
plot(x, time_dif_nurm)
xlabel('z [m]')
ylabel('diff [A.U]')
title('Time diff');

subplot(2,1,2);
plot(x, spectrum_dif_norm)
xlabel('z [m]')
ylabel('diff [A.U]')
title('Spectrum diff');



end