function DM = SimShapeFileDeco2(filename1,filename2,Length_pulse,dt,addphase,Ud,IHz,W1,W2,DM,dec5us)

[power1,phase1]=dataout(filename1,'Test',28,Length_pulse);
[power2,phase2]=dataout(filename2,'Test',28,Length_pulse);

%convert to amp/phase
mxfq1=(1/4/20e-6);
amp1 = power1/100*(mxfq1)*2*pi;
pha1=pi/180*phase1+addphase*pi/2;
mxfq2=(1/4/10e-6);
amp2 = power2/100*(mxfq2)*2*pi;
pha2=pi/180*phase2+addphase*pi/2;
np=Length_pulse;

DM=Ud*DM*Ud';
for n=1:np
    AIz = amp1(n)*IHz(:,1)+amp2(n)*IHz(:,2); PIz = pha1(n)*IHz(:,1)+pha2(n)*IHz(:,2);
    U   = (exp(-1i*PIz).*exp(-1i*PIz)').*(W1*(exp(-1i*dt*AIz).*W2));
    DM  = dec5us.*U*DM*U';
end
DM=Ud*DM*Ud';
end