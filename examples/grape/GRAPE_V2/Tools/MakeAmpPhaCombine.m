function  MakeAmpPhaCombine(InputFile)

    load(InputFile,'-mat','GR','x','NUnitaries','DelayTime');

    
    Ndelay = ceil(DelayTime/GR.del_t)+2*(GR.initdelay/GR.del_t);

    TotalX=GR.Sa+GR.Sp;
    Amp =[]; Pha=[];
    for kk=1:NUnitaries

    AmpTemp = zeros(1,GR.Neach);
    PhaTemp = zeros(1,GR.Neach);
        for j=1:GR.Sa
            if abs(x(j+TotalX*(kk-1),1))>1
                x(j+TotalX*(kk-1),1)=1;
            end
            AmpTemp = AmpTemp + (GR.Amax/GR.Sa)*abs(x(j+TotalX*(kk-1),1))*(sin(2*pi*GR.mAper*x(j+TotalX*(kk-1),2)*GR.t+x(j+TotalX*(kk-1),3))/2 + 1/2);
        end
        AmpTemp=GR.Penal.*AmpTemp;
        
        for j=GR.Sa+1:GR.Sa+GR.Sp
            PhaTemp = PhaTemp + GR.mPha*x(j+TotalX*(kk-1),1)*sin(2*pi*GR.mPper*x(j+TotalX*(kk-1),2).*GR.t+x(j+TotalX*(kk-1),3));
        end
        if kk==NUnitaries
            Amp = [Amp  AmpTemp]; Pha = [Pha  PhaTemp];
        else
            Amp = [Amp  AmpTemp zeros(1,Ndelay(kk))]; Pha = [Pha  PhaTemp zeros(1,Ndelay(kk))];
        end
    end


    u = [transpose(Amp.*cos(Pha)) transpose(Amp.*sin(Pha))];
    Amp=Amp/GR.Amax*100;
    save(InputFile,'Amp','Pha','u','-append')

