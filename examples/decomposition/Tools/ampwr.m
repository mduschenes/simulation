function  ampwr(InputFile)

SavePath = [pwd filesep 'SaveOutputs' filesep 'SaveOutputsGRAPE' filesep];
% SavePath = [pwd filesep 'SaveOutputs' filesep 'SaveOutputsSMOOTH' filesep];
load([SavePath InputFile],'-mat','GR','u');
GR.m=4;
amp=cell(1,GR.m/2);
scaled_amp=cell(1,GR.m/2);
average_power = zeros(1,GR.m/2);
max_power = zeros(1,GR.m/2);
phi=cell(1,GR.m/2);

a1=0:GR.del_t:GR.T; a2=GR.del_t:GR.del_t:GR.T-GR.del_t;
xplot_axis=sort([a1 a2]);

for k=1:GR.m/2
    amp{k}=sqrt((u(:,2*k-1)/(2*pi)).^2 + (u(:,2*k)/(2*pi)).^2);
    scaled_amp{k}=amp{k}*100/(1/4/GR.plength(k));
    
    
    start_amp_jump(1,k) = 0;
    for j = 2:length(scaled_amp{k})
        amp_jump(j) = abs(scaled_amp{k}(j) - scaled_amp{k}(j-1));
        if amp_jump(j) > start_amp_jump(1,k)
            start_amp_jump(1,k)=amp_jump(j);
        end
    end
    start_amp_jump(1,k) = (1/4/GR.plength(k))/100*start_amp_jump(1,k);
    fprintf(2,'Maximum Jump in amplitude for channel %g = %g Khz \n',k,round(start_amp_jump(1,k)*1e-3*1000)/1000) ;
    fprintf(2,'Smoothness for channel %g = %g  \n',k,(sum(abs(amp_jump))/length(amp_jump))) ;
    
    
    
    average_power(k) = sum(scaled_amp{k})/length(scaled_amp{k});
    max_power(k) = max(scaled_amp{k});
    
    
    phi{k}=(atan2(u(:,2*k),u(:,2*k-1)));
    phi{k}=mod(phi{k},2*pi)*180/pi;
    
    phi{k}=reshape(repmat(phi{k},1,2)',1,2*GR.N)';
    scaled_amp{k}=reshape(repmat(scaled_amp{k},1,2)',1,2*GR.N)';
    
    figure
    subplot(2,1,1)
    [AX]=plot(xplot_axis,scaled_amp{k});
    set(get(gca,'Ylabel'),'String','Percentage Power','color',[.9 0 0],'FontSize',12,'FontWeight','bold')
    set(gca,'ycolor',[.9 0 0],'FontSize',12,'FontWeight','bold')
    set(gca,'xcolor',[.9 0 0],'FontSize',12,'FontWeight','bold')
    set(AX,'color',[0.9 0 0],'LineWidth',1.5)
    title(['Avg. Power :',num2str(round(average_power(k)*100)/100),' %  || Max. Power : ',num2str(round(max_power(k)*100)/100),...
        ' %  || Max. Allowed Power : ',num2str(round(1/4/GR.plength(k)*1e-3*100)/100),' KHz'])
    axis tight 
    grid on
    subplot(2,1,2)
    [AY]=plot(xplot_axis,phi{k});
    set(get(gca,'Ylabel'),'String','Phase(Degrees)','color',[ 0 .5 0],'FontSize',12,'FontWeight','bold')
    xlabel(['Time(Seconds) || Channel : ',num2str(k)])
    set(gca,'ycolor',[0 0.5 0],'FontSize',12,'FontWeight','bold')
    set(gca,'xcolor',[0 0.5 0],'FontSize',12,'FontWeight','bold')
    set(AY,'color',[0 0.8 0],'LineWidth',1.5)

    axis tight
    grid on
end
