function GRAPE2bruktest(GRinfo,puls,GRAPE_name,GRAPE_ext,FileOpt)

if nargin<1
    disp('  ')
    disp('      GRAPE2bruk(GRinfo,[P1 P2],''GRAPE_name'',''GRAPE_ext'',''FileOpt'')')
    disp('      Makes BRUKER shape pulsea for the simulated gate')
    disp('  ')
    disp('                     P1,P2 - Time for 90 deg pulse in seconds for nuclei supplied according to spinlist')
    disp('  ')
    disp('            ''GRAPE_name'' - Name for shape file {default is the GRAPEname provided in inputfile}')
    disp(' ')
    disp('             ''GRAPE_ext'' - extenstion for shape file e.g. ''.abc''')
    disp('                                    {default : ''.GRP''}           ')
    disp(' ')
    disp('               ''FileOpt'' - Overwrite or save as a new file?')
    disp('                              ''n'' [for new] / ''o'' [for overwrite] (default ''n''). ')
    disp('                              The output filename is ''GRAPEnameK.mat'', where K is automatically')
    disp('                              selected unless overwriting is opted and GRAPEname is provided in the inputfile') 
    disp(' ')
    disp('      SIMPLEST INPUT : GRAPE2bruk(GRinfo,[P1 P2],[],[])')
    disp('')
    disp('  ')
    disp('   It is unnecessary to provide all the inputs.  Empty inputs indicate')
    disp('   indicate default selections.')
    disp('  ')
    disp('   (Hemant Katiyar, 2012)')
    return
end
 
global gra
gra=GRinfo; 
spinlist=gra.spinlist;

%%%%%%%%%%%%% DEFINING DEFAULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (nargin < 3 | isempty(GRAPE_name)); GRAPE_name = gra.GRAPEname; end;
if (nargin < 4 | isempty(GRAPE_ext)); GRAPE_ext = 'GRP'; end;
if (nargin < 5 | isempty(FileOpt)); FileOpt = 'n'; end;



amp=cell(1,length(spinlist)); phi=cell(1,length(spinlist));
scaled_amp=cell(1,length(spinlist));
N1=cell(1,length(spinlist)); B2=cell(1,length(spinlist));
B1=cell(1,length(spinlist)); N2=cell(1,length(spinlist));

for k=1:length(spinlist)
        amp{k}=sqrt((gra.u(:,k)/(2*pi)).^2 + (gra.u(:,k+length(spinlist))/(2*pi)).^2);
        scaled_amp{k}=100*ones(length(gra.u),1);
        N1{k}=1+floor(log10(scaled_amp{k}));
        B1{k}=scaled_amp{k}./(10.^(N1{k}-1));


%         phi{k}=0*ones(length(gra.u),1);

%%%%%%%%%%% MAKING BRUKER COMPATIBLE %%%%%%%%%%%%%%%%%

%         phi{k}=mod((pi-phi{k}),2*pi);
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% ACTUAL CONVERSION  %%%%%%%%%%%%%%%%%%%%
%         for n=1:length(phi{k})
%             if phi{k}(n)<0
%                phi{k}(n) = phi{k}(n)+2*pi;
%             end
%         end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        phi{k}=0*ones(length(gra.u),1);
%         N2{k}=1+floor(log10(sign(phi{k}).*phi{k}));
%         B2{k}=phi{k}./(10.^(N2{k}-1));
        N2{k}=0*ones(length(gra.u),1);
        B2{k}=90*ones(length(gra.u),1);
end

chlist    = ['ABCDEFGHIJKLMNOPQRSTUVWXYZ'];



for k=1:length(spinlist)
    if (k == 1 && length(spinlist)==1); 
        chname = []; 
    else
        chname = ['_' num2str(length(spinlist)) chlist(k)]; 
    end;
    fileName = [GRAPE_name chname '.' GRAPE_ext];
    
%%%%%%%%%%%%%%%% CHECKING TO PREVENT OVERWRITING FILE %%%%%%%%%%%%%
    if FileOpt == 'n'
         filexist = 0;
         if exist(fileName,'file') == 2
            filexist = 1;
         end
         while filexist > 0; 
            fileName = [GRAPE_name chname num2str(filexist) '.' GRAPE_ext];
            if exist(fileName,'file') == 2
               filexist = filexist + 1;
            else
               filexist = 0;
            end
         end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    fileAddr = ['Bruker_Output' filesep fileName];
    pname = gra.gate;
    minx = min(scaled_amp{k});
    maxx = max(scaled_amp{k});
    miny = min(phi{k});
    maxy = max(phi{k});
    average_power = sum(scaled_amp{k})/length(scaled_amp{k});
    currnt_time=clock;
    fid = fopen(fileAddr,'w');
    fprintf(fid,'##TITLE= %s \n',fileName);
    fprintf(fid,'##$PULSE_NAME= %s\n',pname);
    fprintf(fid,'##JCAMP-DX= 5.00 Bruker JCAMP library \n');
    fprintf(fid,'##DATA TYPE= Shape Data \n');
    fprintf(fid,'##ORIGIN= GRAPE \n');
    fprintf(fid,'##OWNER= <guest> \n');
    fprintf(fid,'##DATE= %s \n',date);
    fprintf(fid,'##TIME(HH/MM/SS)= %g:%g:%0.0f \n',currnt_time(4),currnt_time(5),currnt_time(6));
    fprintf(fid,'##$SHAPE_PARAMETERS= Type: SMP ; Amplitude: 100 \n');
    fprintf(fid,'##--------- SMP PARAMETERS ---------\n');
    fprintf(fid,'##Total_Duration = %5.2f us \n',gra.T*1e6);
    fprintf(fid,'##Initial_Delay = %5.2f us \n',gra.initdelay*1e6);
    fprintf(fid,'##Ending_Delay =%5.2f us \n',gra.initdelay*1e6);
    fprintf(fid,'##Maximum_Amplitude =%g kHz \n',(1/4/puls(k))*1e-3);
    fprintf(fid,'##Transmitter_Offset = %g Hz \n',0.00);
    fprintf(fid,'##----------------------------------\n');
    fprintf(fid,'##MINX= %1.6e \n',minx);
    fprintf(fid,'##MAXX= %1.6e \n',maxx);
    fprintf(fid,'##MINY= %1.6e \n',miny);
    fprintf(fid,'##MAXY= %1.6e \n',maxy);
    fprintf(fid,'##AVG POWER= %1.6e \n',average_power);
    fprintf(fid,'##$SHAPE_EXMODE= Excitation \n');
    fprintf(fid,'##$SHAPE_TOTROT= 9.000000e+01 \n');
    fprintf(fid,'##$SHAPE_BWFAC= 1.116000e+00 \n');
    fprintf(fid,'##$SHAPE_INTEGFAC= 1.000000e+00 \n');
    fprintf(fid,'##$SHAPE_MODE= 1 \n');
    fprintf(fid,'##NPOINTS= %g \n',gra.N);
    fprintf(fid,'##XYPOINTS= (XY..XY) \n');

    for j=1:length(gra.u)
        fprintf(fid,'%5.4fE%0.2i, %5.4fE%0.2i\n',B1{k}(j),N1{k}(j)-1,B2{k}(j),N2{k}(j));
    end
    fprintf(fid,'##END=');
    fclose(fid);
    
    disp(['    The shape file is saved as ' fileAddr]);
    disp('   -------------------------------------------------')
    disp('   ')
end


