function FixPhasesSS(InputFile)

SavePath = [pwd filesep 'SaveOutputs' filesep 'SaveOutputsDecom' filesep];
MainFilesPath = [pwd filesep 'MainPrograms' filesep 'MainProgramsDecom'];
MolPath = [pwd filesep 'MoleculeFiles' filesep ];
addpath(MainFilesPath);

load([SavePath InputFile],'-mat','Par','x','Mol');


for nn=1:Par.nSec
    Ievo = nn*Par.VarPerSec;
    Ilast = (nn-1)*Par.VarPerSec;
    Ph(nn,:) = [x(Ilast+1:Ilast+8-1) ones(1,5)*x(Ilast+8)];
    An(nn,:) = [x(Ilast+8+1:Ilast+2*8-1)  ones(1,5)*x(Ilast+2*8)];
    ti(nn)   = x(Ievo);
end
Ilast=Par.VarPerSec*Par.nSec;
Ph(Par.nSec+1,:)=[x(Ilast+1:Ilast+8-1) ones(1,5)*x(Ilast+8)];
An(Par.nSec+1,:)=[x(Ilast+8+1:Ilast+2*8-1) ones(1,5)*x(Ilast+2*8)];
Anz = [x(Ilast+2*8+1:Ilast+3*8-1) ones(1,5)*x(Ilast+3*8)];

Grad=[];
for n=1:Par.nSec
    Grad=[Grad Ph(n,:) An(n,:) ti(1,n)];
end
Grad =[Grad Ph(n+1,:) An(n+1,:) Anz];
clear x
x=Grad;


run([MolPath Par.MoleculeFile]);
[Ix,Iy,Iz,~,~,~,D] = prodopSparse([1/2 1/2],[7 5]);
Mol.HintFull = genHint(Mol.spinlist,Mol.v,Mol.J,D,Ix,Iy,Iz);
1
Par.VarPerSec=25;
for j=1:Par.nSec
    DelayTime(j)=abs(x(j*Par.VarPerSec))*Par.DelayControl/pi;
end

% Chemical shifts
Mol.v=2*pi*Mol.v;

Phases = transpose(Mol.v).*repmat(DelayTime,Mol.nSpinTotal,1);

PhaseSum = zeros(size(Phases));
for j=1:Par.nSec
    for k=1:j
        PhaseSum(:,j)=PhaseSum(:,j)+Phases(:,k);
    end
end

for n=2:Par.nSec
    Ilast = (n-1)*Par.VarPerSec;
    Atemp=x(Ilast+1:Ilast+Mol.nSpinTotal);
    x(Ilast+1:Ilast+Mol.nSpinTotal)=transpose(transpose(Atemp)+PhaseSum(:,n-1));
end

Ilast=Par.VarPerSec*Par.nSec;
Atemp=x(Ilast+1:Ilast+Mol.nSpinTotal);
x(Ilast+1:Ilast+Mol.nSpinTotal)=transpose(transpose(Atemp)+PhaseSum(:,end));

Btemp=x(Ilast+2*Mol.nSpinTotal+1:Ilast+3*Mol.nSpinTotal);
x(Ilast+2*Mol.nSpinTotal+1:Ilast+3*Mol.nSpinTotal)=transpose(transpose(Btemp)-PhaseSum(:,end));

[~,~,Iz] = prodopSparse(1/2,1);
Had = (1/sqrt(2))*[1 1; 1 -1];
% for j=1:Mol.nSpinTotal-1
%     Had = kron(Had,(1/sqrt(2))*[1 1; 1 -1]);
% end
% O=[1;0]; l=[0;1];
% A=O*O';  B=O*l'; C=l*O';  D=l*l';
% swap=kron(eye(2^6),kron(A,kron(eye(2^4),A)))+kron(eye(2^6),kron(B,kron(eye(2^4),C)))+kron(eye(2^6),kron(C,kron(eye(2^4),B)))+kron(eye(2^6),kron(D,kron(eye(2^4),D)));
% Par.Utarg = swap; 
cnot = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0];
Par.Utarg = kron(cnot,eye(2^10)); 

[Fid,U]=ConstructFinalUnitariesss(x,Mol.nSpinTotal,Par.nSec,Par.VarPerSec,Par.DelayControl,Par.Utarg,Iz,Had,Mol.HintFull);

save([SavePath InputFile 'PC.mat'],'Par','x','Mol','Fid','U','DelayTime')
