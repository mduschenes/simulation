function FixPhases(InputFile)

SavePath = [pwd filesep 'SaveOutputs' filesep 'SaveOutputsDecom' filesep];
MainFilesPath = [pwd filesep 'MainPrograms' filesep 'MainProgramsDecom'];
addpath(MainFilesPath);

load([SavePath InputFile],'-mat','Par','x','Mol');

for j=1:Par.nSec
    DelayTime(j)=abs(x(j*Par.VarPerSec))*Par.DelayControl/pi;
end

% Chemical shifts
Mol.v=2*pi*Mol.v;

Phases = transpose(Mol.v).*repmat(DelayTime,Mol.nSpin,1);

PhaseSum = zeros(size(Phases));
for j=1:Par.nSec
    for k=1:j
        PhaseSum(:,j)=PhaseSum(:,j)+Phases(:,k);
    end
end

for n=2:Par.nSec
    Ilast = (n-1)*Par.VarPerSec;
    Atemp=x(Ilast+1:Ilast+Mol.nSpin);
    x(Ilast+1:Ilast+Mol.nSpin)=transpose(transpose(Atemp)+PhaseSum(:,n-1));
end

Ilast=Par.VarPerSec*Par.nSec;
Atemp=x(Ilast+1:Ilast+Mol.nSpin);
x(Ilast+1:Ilast+Mol.nSpin)=transpose(transpose(Atemp)+PhaseSum(:,end));

Btemp=x(Ilast+2*Mol.nSpin+1:Ilast+3*Mol.nSpin);
x(Ilast+2*Mol.nSpin+1:Ilast+3*Mol.nSpin)=transpose(transpose(Btemp)-PhaseSum(:,end));

[~,~,Iz] = prodopSparse(1/2,Mol.nSpin);
Had = (1/sqrt(2))*[1 1; 1 -1];
for j=1:Mol.nSpin-1
    Had = kron(Had,(1/sqrt(2))*[1 1; 1 -1]);
end
[Fid,U]=ConstructFinalUnitaries(x,Mol.nSpin,Par.nSec,Par.VarPerSec,Par.DelayControl,Par.Utarg,Iz,Had,Mol.HintFull);

save([SavePath InputFile 'PC.mat'],'Par','x','Mol','Fid','U','DelayTime')
