function plotGRAPE(InputFile)

SavePath = [pwd filesep 'SaveOutputs' filesep 'SaveOutputsGRAPE' filesep];
load([SavePath InputFile],'-mat','GR','u','Fid');

a1=0:GR.del_t:GR.T; a2=GR.del_t:GR.del_t:GR.T-GR.del_t;
xplot_axis=sort([a1 a2]);
% GR.m=4;
u=reshape(repmat(u,1,2)',GR.m,2*length(u))';
    
for j=1:GR.m/2
    figure
    hold on
    P1=plot(xplot_axis,u(:,2*j-1)/2/pi/1e3,'r','LineWidth',2);
    P2=plot(xplot_axis,u(:,2*j)/2/pi/1e3,'g','LineWidth',2);
    X1=xlabel('Time(Seconds)');
    Y1=ylabel('RF Power(KHz)');
    T1=title(['GRAPE Controls Variation || Fidelity :',num2str(Fid),' || Channel : ',num2str(j)]);
    set([gca X1 Y1 T1],'FontSize',12,'FontWeight','bold')
    axis tight
    grid on
end




