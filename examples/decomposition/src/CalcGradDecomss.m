function Grad = CalcGradDecomss(x,nSpin,nSec,VarPerSec,DelayControl,Utarg,Ix,Iy,Iz,Hint,Uf,nSub,nSpinT,SubS)

Ub=cell(nSub,2*nSec+2);
for ss=1:nSub
    for kk=1:2*nSec+2
        Ub{ss,kk}=Uf{ss,end}*Uf{ss,kk}';
    end
end

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


EvalGrad = @(x,y,z,a) (2*real(trace(x*y')*z)/2^(2*a));

PhGrad = zeros(nSec+1,nSpinT); 
AnGrad = zeros(nSec+1,nSpinT);
tiGrad = zeros(1,nSec);
AnzGrad= zeros(1,nSpinT);
for ss=1:nSub
    trxjpj = trace(Uf{ss,end}'*Utarg{ss});
    for n=1:nSec   
        for j = 1:nSpin(ss)            
            if n==1
                PhGrad(n,SubS{ss}(j)) = EvalGrad(-1i*(Ub{ss,2*n-1}*(Iz{ss}(:,j).*Uf{ss,2*n-1})-Uf{ss,end}.*transpose(Iz{ss}(:,j))),Utarg{ss},trxjpj,nSpin(ss))+PhGrad(n,SubS{ss}(j));
            else
                PhGrad(n,SubS{ss}(j)) = EvalGrad(-1i*(Ub{ss,2*n-1}*(Iz{ss}(:,j).*Uf{ss,2*n-1})-Ub{ss,2*n-2}*(Iz{ss}(:,j).*Uf{ss,2*n-2})),Utarg{ss},trxjpj,nSpin(ss))+PhGrad(n,SubS{ss}(j));
            end
            AnGrad(n,SubS{ss}(j)) = EvalGrad(-1i*Ub{ss,2*n-1}*(cos(Ph(n,SubS{ss}(j)))*Ix{ss}(:,:,j)+sin(Ph(n,SubS{ss}(j)))*Iy{ss}(:,:,j))*Uf{ss,2*n-1},Utarg{ss},trxjpj,nSpin(ss))+AnGrad(n,SubS{ss}(j));
        end
        tiGrad(1,n) = EvalGrad(-1i*Ub{ss,2*n}*(((ti(n)/abs(ti(n)))*DelayControl/pi*Hint{ss}).*Uf{ss,2*n}),Utarg{ss},trxjpj,nSpin(ss))+tiGrad(1,n);
    end
    
    for j = 1:nSpin(ss)
        PhGrad(nSec+1,SubS{ss}(j)) = EvalGrad(-1i*(Ub{ss,end-1}*(Iz{ss}(:,j).*Uf{ss,end-1})-Ub{ss,end-2}*(Iz{ss}(:,j).*Uf{ss,end-2})),Utarg{ss},trxjpj,nSpin(ss))+PhGrad(n+1,SubS{ss}(j));
        AnGrad(nSec+1,SubS{ss}(j)) = EvalGrad(-1i*Ub{ss,end-1}*(cos(Ph(nSec+1,SubS{ss}(j)))*Ix{ss}(:,:,j)+sin(Ph(nSec+1,SubS{ss}(j)))*Iy{ss}(:,:,j))*Uf{ss,end-1},Utarg{ss},trxjpj,nSpin(ss))+AnGrad(n+1,SubS{ss}(j));
        AnzGrad(SubS{ss}(j)) = EvalGrad(-1i*(Iz{ss}(:,j).*Uf{ss,end}),Utarg{ss},trxjpj,nSpin(ss))+AnzGrad(SubS{ss}(j));
    end
end
PhGrad=PhGrad/nSub;
AnGrad=AnGrad/nSub;
tiGrad=tiGrad/nSub;
AnzGrad=AnzGrad/nSub;

Grad=[];
for n=1:nSec
    Grad=[Grad PhGrad(n,:) AnGrad(n,:) tiGrad(1,n)];
end
Grad =[Grad PhGrad(n+1,:) AnGrad(n+1,:) AnzGrad];