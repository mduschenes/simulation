function [Fid,U] = ConstructFinalUnitariesss(x,nSpin,nSec,...
    VarPerSec,DelayControl,Utarg,Iz,Had,Hint)

USim  = eye(2^nSpin);
U = cell(nSec+1,nSpin);

for n = 1:nSec
    n
    Rxy=1;
    Ilast = (n-1)*VarPerSec;
    Ievo = n*VarPerSec;
    for j=1:nSpin
%         [n,j]
%         [x(Ilast+j),x(Ilast+nSpin+j)]
        PhIz = x(Ilast+j).*Iz;
        AIz  = x(Ilast+nSpin+j).*Iz;
        j
        U{n,j} = (exp(-1i*PhIz).*exp(-1i*PhIz)').*(Had*(exp(-1i*AIz).*Had));
        U{n,j}
        Rxy = kron(Rxy,U{n,j});
    end
    Revo = expm(-1i*abs(x(Ievo))*DelayControl/pi*Hint);
    USim = Revo*(Rxy*USim);
end
Ilast=VarPerSec*nSec;
Rxy=1;
for j=1:nSpin
    PhIz = x(Ilast+j).*Iz;
    AIz  = x(Ilast+nSpin+j).*Iz;
    AzIz = x(Ilast+2*nSpin+j).*Iz;
    U{nSec+1,j} = (exp(-1i*AzIz).*((exp(-1i*PhIz).*exp(-1i*PhIz)').*(Had*(exp(-1i*AIz).*Had))));
    U{nSec+1,j}
    Rxy = kron(Rxy,U{nSec+1,j});
end

USim = (Rxy*USim);

CalcFid = @(x,y,z) abs(trace(x*y'))/2^z;
Fid = CalcFid(USim,Utarg,nSpin);



