function  ampwr(GRinfo,puls)

if nargin<1
    disp('  ')
    disp('      ampwr(GRinfo,[P1 P2])')
    disp('      Calculates Average and Maximum Percentage power and plot the power and phase')
    disp('  ')
    disp('                      P1,P2 - Time for 90 deg pulse in seconds for nuclei supplied according to spinlist')
    disp('  ')
    disp('      SIMPLEST INPUT : ampwr(GRinfo,[p1 p2])')
    disp('')
    disp('  ')
    disp('  ')
    disp('   (Hemant Katiyar, 2012)')
    return
end


global gra
gra=GRinfo; 
spinlist=gra.spinlist;



amp=cell(1,length(spinlist));
scaled_amp=cell(1,length(spinlist));
average_power = zeros(1,length(spinlist));
max_power = zeros(1,length(spinlist));
phi=cell(1,length(spinlist));


for k=1:length(spinlist)
    amp{k}=sqrt((gra.u(:,k)/(2*pi)).^2 + (gra.u(:,k+length(spinlist))/(2*pi)).^2);
    scaled_amp{k}=amp{k}*100/(1/4/puls(k));
    average_power(k) = sum(scaled_amp{k})/length(scaled_amp{k});
    max_power(k) = max(scaled_amp{k});
    
    phi{k}=(atan2(gra.u(:,k+length(spinlist)),gra.u(:,k)));
    phi{k}=mod(phi{k},2*pi)*180/pi;

    figure
    [AX]=plotyy(linspace(0,gra.T,gra.N),scaled_amp{k},linspace(0,gra.T,gra.N),phi{k});
    xlabel(['Time(Seconds) || Channel : ',num2str(k)])
    set(get(AX(1),'Ylabel'),'String','Percentage Power') 
    set(get(AX(2),'Ylabel'),'String','Phase(Degrees)') 
    title(['Avg. Power :',num2str(average_power(k)),' || Max. Power : ',num2str(max_power(k)), ' || Max. Allowed Power : ',num2str(1/4/puls(k)*1e-3),'KHz'])
    grid on
end


