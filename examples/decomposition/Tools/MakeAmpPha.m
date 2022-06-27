function  MakeAmpPha(InputFile)
    load(InputFile,'-mat','GR','x');
    
    Amp = zeros(1,GR.N);
    Pha = zeros(1,GR.N);
    
    for j=1:GR.Sa
        if abs(x(j,1))>1
            x(j,1)=1;
        end
        Amp = Amp + (GR.Amax/GR.Sa)*abs(x(j,1))*(sin(2*pi*GR.mAper*x(j,2)*GR.t+x(j,3))/2 + 1/2);
    end
    
    Amp=GR.Penal.*Amp;
    
    for j=GR.Sa+1:GR.Sa+GR.Sp
        Pha = Pha + GR.mPha*x(j,1)*sin(2*pi*GR.mPper*x(j,2)*GR.t+x(j,3));
    end
    
    u = [transpose(Amp.*cos(Pha)) transpose(Amp.*sin(Pha))];
    Amp=Amp/GR.Amax*100;
    save(InputFile,'Amp','Pha','u','-append')

