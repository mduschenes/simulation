function [fidelity P X] = calculate_robustfidelity_parallel(u)
global gra

clear P X U fidelity

t=(gra.T/gra.N);
Hint=gra.Hint;
Hrf=gra.Hrf;
spinlist = gra.spinlist;
U=zeros(2^gra.nspins,2^gra.nspins,gra.N,length(gra.rfINHrange));
N_spins=gra.nspins;
rfINHrange=gra.rfINHrange;

sum_hamil_dummy=zeros(2^gra.nspins,2^gra.nspins,gra.N,length(gra.rfINHrange));
X=zeros(2^gra.nspins,2^gra.nspins,gra.N,length(gra.rfINHrange));
P=zeros(2^gra.nspins,2^gra.nspins,gra.N,length(gra.rfINHrange));

for k=1:length(gra.rfINHrange)
    
    for j=1:gra.N
        sum_hamil=zeros(2^N_spins);
        for n=1:length(spinlist)
            A=rfINHrange(k)*sqrt(u(j,n)^2+u(j,n+length(spinlist))^2);
            phi=atan2(u(j,n+length(spinlist)),u(j,n));
            sum_hamil = sum_hamil+A*cos(phi)*(Hrf{1,n}) + A*sin(phi)*(Hrf{1,n+length(spinlist)});
        end
        sum_hamil_dummy(:,:,j,k)=sum_hamil;
    end
    
    parfor j=1:gra.N
        U(:,:,j,k) = expm(-1i*t*(Hint + sum_hamil_dummy(:,:,j,k)));
    end
    
    X(:,:,1,k)=eye(2^gra.nspins);
    for j=1:gra.N
        X(:,:,j+1,k)=U(:,:,j,k)*X(:,:,j,k);
    end
    
    P(:,:,gra.N,k)=eye(2^gra.nspins);
    for j=fliplr(2:gra.N)
        P(:,:,j-1,k)=P(:,:,j,k)*U(:,:,j,k);
    end
    fid(k)= (abs(trace(gra.U_target'*X(:,:,gra.N+1,k)))/2^(gra.nspins))^2;
end


fidelity = fid*gra.rfINHiwt;

