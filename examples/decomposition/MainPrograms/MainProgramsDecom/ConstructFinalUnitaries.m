function [Fid,U] = ConstructFinalUnitaries(x,nSpin,nSec,...
                VarPerSec,DelayControl,Utarg,Iz,Had,Hint)

USim  = eye(2^nSpin);
U = cell(1,nSec+1);

for n = 1:nSec
    Ievo = n*VarPerSec;
    Ilast = (n-1)*VarPerSec;
    
    PhIz = sum(repmat(x(Ilast+1:Ilast+nSpin),2^nSpin,1).*Iz,2);
    AIz  = sum(repmat(x(Ilast+nSpin+1:Ilast+2*nSpin),...
           2^nSpin,1).*Iz,2);
    
    Rxy  = (exp(-1i*PhIz).*exp(-1i*PhIz)').*(Had*(exp(-1i*AIz).*Had));
    Revo = expm(-1i*abs(x(Ievo))*DelayControl/pi*Hint);
    
    U{n}=Rxy;
%     U{2*n-1}=Rxy;
%     U{2*n}=Revo;
    
    USim = Revo*(Rxy*USim);
end

Ilast=VarPerSec*nSec;
PhIz = sum(repmat(x(Ilast+1:Ilast+nSpin),2^nSpin,1).*Iz,2);
AIz  = sum(repmat(x(Ilast+nSpin+1:Ilast+2*nSpin),...
       2^nSpin,1).*Iz,2);
AzIz = sum(repmat(x(Ilast+2*nSpin+1:Ilast+3*nSpin),...
       2^nSpin,1).*Iz,2);

Rxy  = (exp(-1i*PhIz).*exp(-1i*PhIz)').*(Had*(exp(-1i*AIz).*Had));
Rz   = exp(-1i*AzIz);

U{nSec+1}=Rz.*Rxy;
% U{1,2*nSec+1}=Rxy;
% U{1,2*nSec+2}=Rz;

USim = Rz.*(Rxy*USim);

CalcFid = @(x,y,z) abs(trace(x*y'))/2^z;
Fid = CalcFid(USim,Utarg,nSpin);



