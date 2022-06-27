function Grad = CalcGradDecom(x,nSpin,nSec,VarPerSec,DelayControl,Utarg,Ix,Iy,Iz,Hint,Uf,D)

Ub=cell(1,2*nSec+2);
for kk=1:2*nSec+2
    Ub{1,kk}=Uf{1,end}*Uf{1,kk}';
end

EvalGrad = @(x,y,z,a) (2*real(trace(x*y')*z)/a^2);

trxjpj = trace(Uf{1,end}'*Utarg);
Grad= zeros(size(x));
for n=1:nSec
    Ievo = n*VarPerSec;
    Ilast = (n-1)*VarPerSec;
    
    for j = 1:nSpin
        if n==1
            Grad(Ilast+j) = EvalGrad(-1i*(Ub{1,2*n-1}*(Iz(:,j).*Uf{1,2*n-1})-Uf{1,end}.*transpose(Iz(:,j))),Utarg,trxjpj,D);
        else
            Grad(Ilast+j) = EvalGrad(-1i*(Ub{1,2*n-1}*(Iz(:,j).*Uf{1,2*n-1})-Ub{1,2*n-2}*(Iz(:,j).*Uf{1,2*n-2})),Utarg,trxjpj,D);
        end
        Grad(Ilast+nSpin+j) = EvalGrad(-1i*Ub{1,2*n-1}*(cos(x(Ilast+j))*Ix(:,:,j)+sin(x(Ilast+j))*Iy(:,:,j))*Uf{1,2*n-1},Utarg,trxjpj,D);
    end
    Grad(Ievo) = EvalGrad(-1i*Ub{1,2*n}*(((x(Ievo)/abs(x(Ievo)))*DelayControl/pi*Hint).*Uf{1,2*n}),Utarg,trxjpj,D);
end

Ilast=VarPerSec*nSec;
for j = 1:nSpin
    Grad(Ilast+j) = EvalGrad(-1i*(Ub{1,end-1}*(Iz(:,j).*Uf{1,end-1})-Ub{1,end-2}*(Iz(:,j).*Uf{1,end-2})),Utarg,trxjpj,D);
    Grad(Ilast+nSpin+j) = EvalGrad(-1i*Ub{1,end-1}*(cos(x(Ilast+j))*Ix(:,:,j)+sin(x(Ilast+j))*Iy(:,:,j))*Uf{1,end-1},Utarg,trxjpj,D); 
    Grad(Ilast+2*nSpin+j) = EvalGrad(-1i*(Iz(:,j).*Uf{1,end}),Utarg,trxjpj,D);
end

