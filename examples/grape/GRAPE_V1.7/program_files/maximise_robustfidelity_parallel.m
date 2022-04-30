function [epsi_new u skipping_calc_fid U_s X_s] = maximise_robustfidelity_parallel(P,X,u,epsi,iter,Fid_first)
global gra

t=(gra.T/gra.N);
U_t=gra.U_target;
Hrf=gra.Hrf;
Hint=gra.Hint;
epsi_range=[0 epsi 2*epsi];
spinlist = gra.spinlist;

su=zeros(gra.N,gra.m);
Number_Of_Steps=gra.N;
Number_Of_Controls=gra.m;
Number_Of_Spins=gra.nspins;
RFinhomo_range=gra.rfINHrange;


for l=1:length(gra.rfINHrange)
    trxjpj = trace(X(:,:,gra.N+1,l)'*U_t);
    for k=1:Number_Of_Controls
        Hrf_dummy=Hrf{1,k};
        parfor j=1:Number_Of_Steps
            XjUtPj = X(:,:,j+1,l)*U_t'*P(:,:,j,l);
            af_dummy(j,k,l) = -2*real(trace(XjUtPj*1i*t*Hrf_dummy)*trxjpj)/2^(2*Number_Of_Spins);
        end
    end
    gra.af=af_dummy;
    su=su+gra.rfINHiwt(l)*gra.af(:,:,l);
    gra.af=su;
end


lambda=find_lambda_bconst(iter);

U=zeros(2^gra.nspins,2^gra.nspins,gra.N,length(gra.rfINHrange));
sum_hamil_dummy=zeros(2^gra.nspins,2^gra.nspins,gra.N,length(gra.rfINHrange));

fide(1)=Fid_first;
for d=2:length(epsi_range)
    u_new = u + epsi_range(d)*lambda;
    u_new=penalizecontrols(u_new);
    fid=zeros(1,length(gra.rfINHrange));
    for k=1:length(gra.rfINHrange)
        for j=1:Number_Of_Steps
            sum_hamil=zeros(2^Number_Of_Spins);
            for n=1:length(spinlist)
                A=RFinhomo_range(k)*sqrt(u_new(j,n)^2+u_new(j,n+length(spinlist))^2);
                phi=atan2(u_new(j,n+length(spinlist)),u_new(j,n));
                sum_hamil = sum_hamil+A*cos(phi)*(Hrf{1,n}) + A*sin(phi)*(Hrf{1,n+length(spinlist)});
            end
            sum_hamil_dummy(:,:,j,k)=sum_hamil;
        end
        parfor j=1:Number_Of_Steps
            U(:,:,j,k) = expm(-1i*t*(Hint + sum_hamil_dummy(:,:,j,k)));
        end
        eval(['Usave' num2str(d) '=U;']);
        X(:,:,1,k)=eye(2^gra.nspins);
        for j=1:Number_Of_Steps
            X(:,:,j+1,k)=U(:,:,j,k)*X(:,:,j,k);
        end
        eval(['Xsave' num2str(d) '=X;']);
        fid(k)= (abs(trace(gra.U_target'*X(:,:,gra.N+1,k)))/2^(gra.nspins))^2;
    end
    fide(d) = fid*gra.rfINHiwt;
end

multi_fac=quadratic_fit(fide);
gra.mfa(iter)=multi_fac;
epsi_new=multi_fac*epsi;
u=u+epsi_new*lambda;

u=penalizecontrols(u);

if(multi_fac==2)
    skipping_calc_fid = 1;
    U_s=Usave3;
    X_s=Xsave3;
elseif(multi_fac==1)
    skipping_calc_fid = 1;
    U_s=Usave2;
    X_s=Xsave2;
else
    skipping_calc_fid = 0;
    U_s=[];
    X_s=[];
end
