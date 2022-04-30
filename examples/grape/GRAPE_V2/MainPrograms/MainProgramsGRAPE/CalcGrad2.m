function Grad = CalcGrad2(Uf,nspins,rfi,IHX,IHY,N,m,UTopt,delT,GrPenal,W)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Backward propagator %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ub=cell(N,size(rfi,1));
for l=1:size(rfi,1)
    for j=1:N
        Ub{j,l} = Uf{end,l}*Uf{j,l}';
    end
end

EvalGrad = @(x,z,a) (2*real(trace(x)*z)/2^(2*a));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gradient Calculation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Grad=zeros(N,m);
for l=1:size(rfi,1)
    trxjpj = trace(Uf{end,l}'*UTopt{l});
    for j=1:N
        XX=Uf{j,l}*UTopt{l}';
        for k=1:m/2
            Grad(j,2*k)   = Grad(j,2*k)+rfi(l,2)*(EvalGrad((Ub{j,l}*IHY{k})*XX,(-1i*delT)*trxjpj,nspins)-W*(GrPenal(j,2*k)));
            Grad(j,2*k-1) = Grad(j,2*k-1)+rfi(l,2)*(EvalGrad((Ub{j,l}*IHX{k})*XX,(-1i*delT)*trxjpj,nspins)-W*(GrPenal(j,2*k-1)));
        end
    end
end
