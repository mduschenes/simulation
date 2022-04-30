function  ampwrNOPLOT(InputFile)

SavePath = [pwd filesep 'SaveOutputs' filesep 'SaveOutputsGRAPE' filesep];
% SavePath = [pwd filesep 'SaveOutputs' filesep 'SaveOutputsSMOOTH' filesep];
load([SavePath InputFile],'-mat','GR','u');

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
end
