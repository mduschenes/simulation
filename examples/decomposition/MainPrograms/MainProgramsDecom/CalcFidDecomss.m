function [Fid,Uf,Fids] = CalcFidDecomss(x,nSpin,nSec,VarPerSec,DelayControl,Iz,Had,Hint,Utarg,nSub,nSpinT,SubS)

% Function Definition
CalcFid = @(x,y,z) abs(trace(x*y'))/2^z;

for n = 1:nSec
    Ievo = n*VarPerSec;
    Ilast = (n-1)*VarPerSec;

    Ph(n,:)= (x(Ilast+1:Ilast+nSpinT));
    An(n,:)= (x(Ilast+nSpinT+1:Ilast+2*nSpinT));
    ti(n)  = x(Ievo);
    
end
Ilast=VarPerSec*nSec;
Ph(nSec+1,:)=(x(Ilast+1:Ilast+nSpinT));
An(nSec+1,:)=(x(Ilast+nSpinT+1:Ilast+2*nSpinT));
Anz = (x(Ilast+2*nSpinT+1:Ilast+3*nSpinT));

Uf  = cell(nSub,2*nSec+2);
for ss=1:nSub
    for n = 1:nSec
        PhIz = sum(repmat(Ph(n,SubS{ss}),2^nSpin(ss),1).*Iz{ss},2);
        AIz  = sum(repmat(An(n,SubS{ss}),2^nSpin(ss),1).*Iz{ss},2);
        
        Rxy  = (exp(-1i*PhIz).*exp(-1i*PhIz)').*(Had{ss}*(exp(-1i*AIz).*Had{ss}));
        Revo = exp(-1i*abs(ti(n))*DelayControl/pi*Hint{ss});
        
        if n==1
            Uf{ss,2*n-1} = Rxy;
        else
            Uf{ss,2*n-1} = Rxy*Uf{ss,2*n-2};
        end
        Uf{ss,2*n}   = Revo.*Uf{ss,2*n-1};
    end
    
    PhIz = sum(repmat(Ph(nSec+1,SubS{ss}),2^nSpin(ss),1).*Iz{ss},2);
    AIz  = sum(repmat(An(nSec+1,SubS{ss}),2^nSpin(ss),1).*Iz{ss},2);
    AzIz = sum(repmat(Anz(SubS{ss}),2^nSpin(ss),1).*Iz{ss},2);

    Rxy  = (exp(-1i*PhIz).*exp(-1i*PhIz)').*(Had{ss}*(exp(-1i*AIz).*Had{ss}));
    Rz   = exp(-1i*AzIz);
    
    Uf{ss,2*nSec+1}=Rxy*Uf{ss,2*nSec};
    Uf{ss,2*nSec+2}=Rz.*Uf{ss,2*nSec+1};
%     size(Uf{ss,end})
%     size(Utarg{ss})
    Fids(ss) = CalcFid(Uf{ss,end},Utarg{ss},nSpin(ss));
end
Fid = sum(Fids)/nSub;



