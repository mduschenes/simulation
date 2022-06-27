function [Fid,Uf,Fids] = CalcFidDecomss12(x,nSpin,nSec,VarPerSec,DelayControl,Iz,Had,Hint,Utarg,nSub,nSpinT,SubS)

% Function Definition
CalcFid = @(x,y,z) abs(trace(x*y'))/2^z;

for n = 1:nSec
    Ievo = n*VarPerSec;
    Ilast = (n-1)*VarPerSec;

    Ph(n,:)= (x(Ilast+1:Ilast+8));
    An(n,:)= (x(Ilast+8+1:Ilast+2*8));
    ti(n)  = x(Ievo);
    
end
Ilast=VarPerSec*nSec;
Ph(nSec+1,:)=(x(Ilast+1:Ilast+8));
An(nSec+1,:)=(x(Ilast+8+1:Ilast+2*8));
Anz = (x(Ilast+2*8+1:Ilast+3*8));

Uf  = cell(nSub,2*nSec+2);
for ss=1:nSub
    for n = 1:nSec
        for nn = 1:length(SubS{ss}); if SubS{ss}(nn)>=8 && SubS{ss}(nn)<=12; SubS{ss}(nn)=8; end;end 
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
    
    Fids(ss) = CalcFid(Uf{ss,end},Utarg{ss},nSpin(ss));
end

Fid = sum(Fids)/nSub;

% function [Fid,Uf,Fids] = CalcFidDecomss12(x,nSpin,nSec,VarPerSec,DelayControl,Iz,Had,Hint,Utarg,nSub,nSpinT,SubS,V)
% 
% % Function Definition
% CalcFid = @(x,y,z) abs(trace(x*y'))/2^z;
% 
% V=2*pi*V;
% 
% for n = 1:nSec
%     Ievo = n*VarPerSec;
%     Ilast = (n-1)*VarPerSec;
% 
%     Ph(n,:)= (x(Ilast+1:Ilast+8));
%     An(n,:)= (x(Ilast+8+1:Ilast+2*8));
%     ti(n)  = x(Ievo);
%     
% end 
% Ilast=VarPerSec*nSec;
% Ph(nSec+1,:)=(x(Ilast+1:Ilast+8));
% An(nSec+1,:)=(x(Ilast+8+1:Ilast+2*8));
% Anz = (x(Ilast+2*8+1:Ilast+3*8));
% 
% Uf  = cell(nSub,2*nSec+2);
% for ss=1:nSub
%     TrueSubS = SubS{ss};
%     for n = 1:nSec
%         for nn = 1:length(SubS{ss}); if SubS{ss}(nn)>=8 && SubS{ss}(nn)<=12; SubS{ss}(nn)=8; end;end
%         if n==1
%             PhIz = sum(repmat(Ph(n,SubS{ss}),2^nSpin(ss),1).*Iz{ss},2);
%         else
%             VV=[]; for jjj=1:length(TrueSubS); if TrueSubS(jjj)>=8; VV(jjj)=V(TrueSubS(jjj)); else; VV(jjj)=0; end; end
%             PhIz = sum(repmat(Ph(n,SubS{ss})-VV*sum(abs(ti(1:n-1))*DelayControl/pi),2^nSpin(ss),1).*Iz{ss},2);
%         end
%         AIz  = sum(repmat(An(n,SubS{ss}),2^nSpin(ss),1).*Iz{ss},2);
%         
%         Rxy  = (exp(-1i*PhIz).*exp(-1i*PhIz)').*(Had{ss}*(exp(-1i*AIz).*Had{ss}));
%         Revo = exp(-1i*abs(ti(n))*DelayControl/pi*Hint{ss});
%         
%         if n==1
%             Uf{ss,2*n-1} = Rxy;
%         else
%             Uf{ss,2*n-1} = Rxy*Uf{ss,2*n-2};
%         end
%         Uf{ss,2*n}   = Revo.*Uf{ss,2*n-1};
%     end
%     VV=[]; for jjj=1:length(TrueSubS); if TrueSubS(jjj)>=8; VV(jjj)=V(TrueSubS(jjj)); else; VV(jjj)=0; end; end
%     PhIz = sum(repmat(Ph(nSec+1,SubS{ss})-VV*sum(abs(ti(1:nSec))*DelayControl/pi),2^nSpin(ss),1).*Iz{ss},2);
%     AIz  = sum(repmat(An(nSec+1,SubS{ss}),2^nSpin(ss),1).*Iz{ss},2);
%     AzIz = sum(repmat(Anz(SubS{ss})+VV*sum(abs(ti(1:nSec))*DelayControl/pi),2^nSpin(ss),1).*Iz{ss},2);
% 
%     Rxy  = (exp(-1i*PhIz).*exp(-1i*PhIz)').*(Had{ss}*(exp(-1i*AIz).*Had{ss}));
%     Rz   = exp(-1i*AzIz);
%     
%     Uf{ss,2*nSec+1}=Rxy*Uf{ss,2*nSec};
%     Uf{ss,2*nSec+2}=Rz.*Uf{ss,2*nSec+1};
%     
%     Fids(ss) = CalcFid(Uf{ss,end},Utarg{ss},nSpin(ss));
% end
% 
% Fid = sum(Fids)/nSub;
