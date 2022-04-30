function [uprange,exitflag] = penalty(div,plength,askplot)

global gra

if length(div)==1
    for j=2:gra.m/2
        div(j)=div(1);
    end
end

uprange=zeros(gra.N,1,gra.m/2);

for j=1:gra.m/2
    per_div(j)=(div(j)*gra.N)/100;
    x=linspace(1,gra.N/per_div(j),gra.N); x=x-mean(x);
    penal=cosh(x')-1; penal=penal/max(penal);
    Amax=(1/4/plength(j))*2*pi*99/100;
    uprange(:,:,j) = -Amax*(penal);  
    uprange(:,:,j) = uprange(:,:,j)-min(uprange(:,:,j));
end


exitflag=0;
if askplot=='y'
    for j=1:gra.m/2
        figure
        plot(uprange(:,:,j))
    end
    reply = input('Do you want to continue? Y/N: ', 's');
    if (reply=='n' || isempty(reply))
       exitflag=1;
    end
end