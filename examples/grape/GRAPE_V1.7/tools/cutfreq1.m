function trunc_unew = cutfreq1(u,per)

if(per>50) ; error('Percentage is more than 50 % all entries will be zeros'); end

unew = u;

funew = fftshift(fft(unew));
num2cut = round(length(funew)*per/100);

% for j=1:2
    funew(1:num2cut,1) = 0;
%     funew(length(funew)-num2cut:end,j)=0;
% end
ifunew = ifft(ifftshift(funew));

trunc_unew = real(ifunew);