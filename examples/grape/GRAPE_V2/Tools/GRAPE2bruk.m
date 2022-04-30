function GRAPE2bruk(InputFile,GRAPE_name,GRAPE_ext,FileTransfer,FileOpt)

SavePath = [pwd filesep 'SaveOutputs' filesep 'SaveOutputsGRAPE' filesep];
load([SavePath InputFile],'-mat','GR','u');

%%%%%%%%%%%%% DEFINING DEFAULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (nargin < 2 || isempty(GRAPE_name)); GRAPE_name = GR.GRAPEname; end
if (nargin < 3 || isempty(GRAPE_ext)); GRAPE_ext = 'hem'; end
if (nargin < 4 || isempty(FileTransfer)); FileTransfer = 0; end
if (nargin < 5 || isempty(FileOpt)); FileOpt = 'n'; end

amp=cell(1,GR.m/2); phi=cell(1,GR.m/2);
scaled_amp=cell(1,GR.m/2);

for k=1:GR.m/2
    amp{k}=sqrt((u(:,2*k-1)/(2*pi)).^2 + (u(:,2*k)/(2*pi)).^2);
    scaled_amp{k}=amp{k}*100/(1/4/GR.plength(k));

    phi{k}=(atan2(u(:,2*k),u(:,2*k-1)));
    %%%%%%%%%%%%%% ACTUAL CONVERSION  %%%%%%%%%%%%%%%%%%%%
    for n=1:length(phi{k})
        if phi{k}(n)<0
            phi{k}(n) = phi{k}(n)+2*pi;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    phi{k}=phi{k}*180/pi;
end

chlist    = ['CHCDEFGHIJKLMNOPQRSTUVWXYZ'];

for k=1:GR.m/2
    if (k == 1 && GR.m/2==1)
        chname = [];
    else
%         chname = ['_' num2str(GR.m/2) chlist(k)];
        chname = ['_' chlist(k)];
    end;
    fileName = [GRAPE_name chname '.' GRAPE_ext];
    %     fileName = [GRAPE_name chname];
    
    %%%%%%%%%%%%%%%% CHECKING TO PREVENT OVERWRITING FILE %%%%%%%%%%%%%
    if FileOpt == 'n'
        filexist = 0;
        if exist(fileName,'file') == 2
            filexist = 1;
        end
        while filexist > 0;
            fileName = [GRAPE_name chname num2str(filexist) '.' GRAPE_ext];
            %               fileName = [GRAPE_name chname num2str(filexist)];
            if exist(fileName,'file') == 2
                filexist = filexist + 1;
            else
                filexist = 0;
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    fileAddr = ['BrukerFiles' filesep fileName];
    minx = min(scaled_amp{k});
    maxx = max(scaled_amp{k});
    miny = min(phi{k});
    maxy = max(phi{k});
    average_power = sum(scaled_amp{k})/length(scaled_amp{k});
    currnt_time=clock;
    fid = fopen(fileAddr,'w');
    fprintf(fid,'##TITLE= %s \n',fileName);
    fprintf(fid,'##JCAMP-DX= 5.00 Bruker JCAMP library \n');
    fprintf(fid,'##DATA TYPE= Shape Data \n');
    fprintf(fid,'##ORIGIN= GRAPE \n');
    fprintf(fid,'##OWNER= <guest> \n');
    fprintf(fid,'##DATE= %s \n',date);
    fprintf(fid,'##TIME(HH/MM/SS)= %g:%g:%0.0f \n',currnt_time(4),currnt_time(5),currnt_time(6));
    fprintf(fid,'##$SHAPE_PARAMETERS= Type: SMP ; Amplitude: 100 \n');
    fprintf(fid,'##--------- SMP PARAMETERS ---------\n');
    fprintf(fid,'##Total_Duration = %5.2f us \n',GR.T*1e6);
    fprintf(fid,'##Initial_Delay = %5.2f us \n',GR.initdelay*1e6);
    fprintf(fid,'##Ending_Delay =%5.2f us \n',GR.initdelay*1e6);
    fprintf(fid,'##Maximum_Amplitude =%g kHz \n',(1/4/GR.plength(k))*1e-3);
    fprintf(fid,'##Transmitter_Offset = %g Hz \n',0.00);
    fprintf(fid,'##----------------------------------\n');
    fprintf(fid,'##MINX= %1.6e \n',minx);
    fprintf(fid,'##MAXX= %1.6e \n',maxx);
    fprintf(fid,'##MINY= %1.6e \n',miny);
    fprintf(fid,'##MAXY= %1.6e \n',maxy);
    fprintf(fid,'##AVG POWER= %1.6e \n',average_power);
    fprintf(fid,'##$SHAPE_EXMODE= None \n');
    fprintf(fid,'##$SHAPE_TOTROT= %7.6e\n',90);
    fprintf(fid,'##$SHAPE_BWFAC= %7.6e\n',1);
    fprintf(fid,'##$SHAPE_INTEGFAC= %7.6e\n',1);
    fprintf(fid,'##$SHAPE_MODE= 1 \n');
    fprintf(fid,'##NPOINTS= %g \n',GR.N);
    fprintf(fid,'##XYPOINTS= (XY..XY) \n');
    
    for j=1:GR.N
        fprintf(fid,'  %7.6e,  %7.6e\n',scaled_amp{k}(j),phi{k}(j));
    end
    fprintf(fid,'##END=');
    fclose(fid);
    
    disp(['    The shape file is saved as ' fileAddr]);
    disp('   -------------------------------------------------')
    disp('   ')
    if (maxx > 100)
        disp('Maximum power is more than 100')
    end
end
fclose('all');

if FileTransfer==1
    load StructSCP.mat
    [scp_put(StructSCP, localFilename, remotePath, localPath, remoteFilename)]
end

