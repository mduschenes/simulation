function trunc_unew = cutfreq(GRinfo,per)

if nargin<1
    disp('  ')
    disp('      trunc_unew = cutfreq(GRinfo,per)')
    disp('      Truncates the higher frequency terms')
    disp('  ')
    disp('                      per - percentage of high frequency components to truncate')
    disp('  ')
    disp('      SIMPLEST INPUT : trunc_unew = cutfreq(GRinfo,per)')
    disp('  ')
    disp('  ')
    disp('  ')
    disp('   (Hemant Katiyar, 2012)')
    return
end


global gra
gra=GRinfo;

if(per>50) ; error('Percentage is more than 50 % all entries will be zeros'); end

unew = gra.u;

funew = fftshift(fft(unew));
num2cut = round(length(funew)*per/100);

for j=1:gra.m
    funew(1:num2cut,j) = 0;
    funew(length(funew)-num2cut:end,j)=0;
end
ifunew = ifft(ifftshift(funew));

trunc_unew = real(ifunew);