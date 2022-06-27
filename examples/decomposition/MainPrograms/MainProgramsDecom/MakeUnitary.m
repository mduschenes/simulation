function U = MakeUnitary(x,nSpin,nSec,VarPerSec,DelayControl,Iz,Had,Hint)

U = eye(2^nSpin);

for n = 1:nSec
    Ievo = n*VarPerSec;
    Ilast = (n-1)*VarPerSec;

    PhIz = sum(repmat(x(Ilast+1:Ilast+nSpin),2^nSpin,1).*Iz,2);
    AIz  = sum(repmat(x(Ilast+nSpin+1:Ilast+2*nSpin),2^nSpin,1).*Iz,2);

    Rxy  = (exp(-1i*PhIz).*exp(-1i*PhIz)').*(Had*(exp(-1i*AIz).*Had));
    Revo =  exp(-1i*abs(x(Ievo))*DelayControl/pi*Hint);

    U=Revo.*Rxy*U;
end

Ilast=VarPerSec*nSec;
PhIz = sum(repmat(x(Ilast+1:Ilast+nSpin),2^nSpin,1).*Iz,2);
AIz  = sum(repmat(x(Ilast+nSpin+1:Ilast+2*nSpin),2^nSpin,1).*Iz,2);
AzIz = sum(repmat(x(Ilast+2*nSpin+1:Ilast+3*nSpin),2^nSpin,1).*Iz,2);

Rxy  = (exp(-1i*PhIz).*exp(-1i*PhIz)').*(Had*(exp(-1i*AIz).*Had));
Rz   = exp(-1i*AzIz);

U=Rz.*Rxy*U;
