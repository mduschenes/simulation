function [Fid,U,Usec] = ConstructFinalUnitaries2(x,nSpin,nSec,...
                VarPerSec,DelayControl,Utarg,Hint)

USim  = eye(2^nSpin);
U = cell(nSec+1,nSpin);
Usec = cell(1,nSec+1);

[~,~,Iz]=prodopSparse(1/2,1);
Had=[1 1; 1 -1]/sqrt(2);

for n = 1:nSec
    Ievo = n*VarPerSec;
    Ilast = (n-1)*VarPerSec;
    
    Ph=x(Ilast+1:Ilast+nSpin);
    An=x(Ilast+nSpin+1:Ilast+2*nSpin);
    
    Rxy=1;
    for k=1:nSpin
        PhIz = Ph(k)*Iz; AIz = An(k)*Iz;
        U{n,k}=(exp(-1i*PhIz).*exp(-1i*PhIz)').*(Had*(exp(-1i*AIz).*Had));
        Rxy = kron(Rxy,U{n,k});
    end
    Revo = expm(-1i*abs(x(Ievo))*DelayControl/pi*Hint);
    
    Usec{n}=Rxy;
    USim = Revo*(Rxy*USim);
end

Ilast=VarPerSec*nSec;

Ph=x(Ilast+1:Ilast+nSpin);
An=x(Ilast+nSpin+1:Ilast+2*nSpin);
Anz=x(Ilast+2*nSpin+1:Ilast+3*nSpin);
Rxyz=1;
for k=1:nSpin
    PhIz = Ph(k)*Iz; AIz = An(k)*Iz; AzIz=Anz(k)*Iz;
    U{nSec+1,k}=exp(-1i*AzIz).*((exp(-1i*PhIz).*exp(-1i*PhIz)').*(Had*(exp(-1i*AIz).*Had)));
    Rxyz = kron(Rxyz,U{nSec+1,k});
end
Usec{nSec+1}=Rxyz;
USim = (Rxyz*USim);

CalcFid = @(x,y,z) abs(trace(x*y'))/2^z;
Fid = CalcFid(USim,Utarg,nSpin);



