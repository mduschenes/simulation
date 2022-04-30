function [epsi_new u] = maximise_robustfidelity_state(P,X,u,epsi,iter)
global gra

t=(gra.T/gra.N);
Hrf=gra.Hrf;
epsi_range=[0 epsi 2*epsi];
spinlist = gra.spinlist;

su=zeros(gra.N,gra.m);
for l=1:length(gra.rfINHrange)
    for j=1:gra.N
        for k=1:gra.m
            gra.af(j,k,l) = -2*real(trace(P(:,:,j,l)'*gra.RHO_targ*P(:,:,j,l)*1i*t*(Hrf{1,k}*X(:,:,j+1,l)*gra.RHO_init*X(:,:,j+1,l)'-X(:,:,j+1,l)*gra.RHO_init*X(:,:,j+1,l)'*Hrf{1,k})));
        end
    end
    su=su+gra.rfINHiwt(l)*gra.af(:,:,l);
    gra.af=su;
end


lambda=find_lambda_bconst(iter);

U=zeros(2^gra.nspins,2^gra.nspins,gra.N);

for d=1:length(epsi_range)
    u_new = u + epsi_range(d)*lambda;
    fid=zeros(1,length(gra.rfINHrange));
    for k=1:length(gra.rfINHrange)
        X(:,:,1)=eye(2^gra.nspins);
        for j=1:gra.N
            sum_hamil=zeros(2^gra.nspins);
            for n=1:length(spinlist)
                A=gra.rfINHrange(k)*sqrt(u_new(j,n)^2+u_new(j,n+length(spinlist))^2);
                phi=atan2(u_new(j,n+length(spinlist)),u_new(j,n));
                sum_hamil = sum_hamil+A*cos(phi)*(gra.Hrf{1,n}) + A*sin(phi)*(gra.Hrf{1,n+length(spinlist)});
            end
            U(:,:,j,k) = expm(-1i*(gra.T/gra.N)*(gra.Hint + sum_hamil));
            X(:,:,j+1,k)=U(:,:,j,k)*X(:,:,j,k);
        end
        fid(k)= (abs(trace(gra.RHO_targ'*X(:,:,gra.N+1,k)*gra.RHO_init*X(:,:,gra.N+1,k)')));
    end
    fide(d) = fid*gra.rfINHiwt;
end

multi_fac=quadratic_fit(fide);
gra.mfa(iter)=multi_fac;
epsi_new=multi_fac*epsi;
u=u+epsi_new*lambda;

u=penalizecontrols(u);

