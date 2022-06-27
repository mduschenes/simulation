function [U,Grad] = MakeUnitaryAndGrad(x,nSpin,nSec,VarPerSec,DelayControl,Utarg,Ix,Iy,Iz,Had,Hint)

Uf  = cell(1,2*nSec+2);
for n = 1:nSec
    Ievo = n*VarPerSec;
    Ilast = (n-1)*VarPerSec;

    PhIz = sum(repmat(x(Ilast+1:Ilast+nSpin),2^nSpin,1).*Iz,2);
    AIz  = sum(repmat(x(Ilast+nSpin+1:Ilast+2*nSpin),2^nSpin,1).*Iz,2);

    Rxy  = (exp(-1i*PhIz).*exp(-1i*PhIz)').*(Had*(exp(-1i*AIz).*Had));
    Revo = exp(-1i*abs(x(Ievo))*DelayControl/pi*Hint);

    if n==1
        Uf{1,2*n-1} = Rxy;
    else
        Uf{1,2*n-1} = Rxy*Uf{1,2*n-2};
    end
    Uf{1,2*n}   = Revo.*Uf{1,2*n-1};    
end

Ilast=VarPerSec*nSec;
PhIz = sum(repmat(x(Ilast+1:Ilast+nSpin),2^nSpin,1).*Iz,2);
AIz  = sum(repmat(x(Ilast+nSpin+1:Ilast+2*nSpin),2^nSpin,1).*Iz,2);
AzIz = sum(repmat(x(Ilast+2*nSpin+1:Ilast+3*nSpin),2^nSpin,1).*Iz,2);

Rxy  = (exp(-1i*PhIz).*exp(-1i*PhIz)').*(Had*(exp(-1i*AIz).*Had));
Rz   = exp(-1i*AzIz);

Uf{1,2*nSec+1}=Rxy*Uf{1,2*nSec};
Uf{1,2*nSec+2}=Rz.*Uf{1,2*nSec+1};

U = Uf{1,end};


Ub=cell(1,2*nSec+2);
for kk=1:2*nSec+2
    Ub{1,kk}=Uf{1,end}*Uf{1,kk}';
end

EvalGrad = @(x,y,z,a) (2*real(trace(x*y')*z)/2^(2*a));

trxjpj = trace(U'*Utarg);
for n=1:nSec
    Ievo = n*VarPerSec;
    Ilast = (n-1)*VarPerSec;
    
    for j = 1:nSpin
        if n==1
            Grad(Ilast+j) = EvalGrad(-1i*(Ub{1,2*n-1}*(Iz(:,j).*Uf{1,2*n-1})-Uf{1,end}.*transpose(Iz(:,j))),Utarg,trxjpj,nSpin);
        else
            Grad(Ilast+j) = EvalGrad(-1i*(Ub{1,2*n-1}*(Iz(:,j).*Uf{1,2*n-1})-Ub{1,2*n-2}*(Iz(:,j).*Uf{1,2*n-2})),Utarg,trxjpj,nSpin);
        end
        Grad(Ilast+nSpin+j) = EvalGrad(-1i*Ub{1,2*n-1}*(cos(x(Ilast+j))*Ix(:,:,j)+sin(x(Ilast+j))*Iy(:,:,j))*Uf{1,2*n-1},Utarg,trxjpj,nSpin);
    end
    Grad(Ievo) = EvalGrad(-1i*Ub{1,2*n}*(((x(Ievo)/abs(x(Ievo)))*DelayControl/pi*Hint).*Uf{1,2*n}),Utarg,trxjpj,nSpin);
end

Ilast=VarPerSec*nSec;
for j = 1:nSpin
    Grad(Ilast+j) = EvalGrad(-1i*(Ub{1,end-1}*(Iz(:,j).*Uf{1,end-1})-Ub{1,end-2}*(Iz(:,j).*Uf{1,end-2})),Utarg,trxjpj,nSpin);
    Grad(Ilast+nSpin+j) = EvalGrad(-1i*Ub{1,end-1}*(cos(x(Ilast+j))*Ix(:,:,j)+sin(x(Ilast+j))*Iy(:,:,j))*Uf{1,end-1},Utarg,trxjpj,nSpin); 
    Grad(Ilast+2*nSpin+j) = EvalGrad(-1i*(Iz(:,j).*Uf{1,end}),Utarg,trxjpj,nSpin);
end

