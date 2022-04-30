function [fidelity P X] = calculate_robustfidelity_state(u)
global gra

clear P X U fidelity

spinlist = gra.spinlist;
U=zeros(2^gra.nspins,2^gra.nspins,gra.N,length(gra.rfINHrange));
A=zeros(gra.N,length(spinlist));
phi=zeros(gra.N,length(spinlist));


for k=1:length(gra.rfINHrange)
    X(:,:,1,k)=eye(2^gra.nspins);
    for j=1:gra.N
        sum_hamil=zeros(2^gra.nspins);
        for n=1:length(spinlist)
            A=gra.rfINHrange(k)*sqrt(u(j,n)^2+u(j,n+length(spinlist))^2);
            phi=atan2(u(j,n+length(spinlist)),u(j,n));
            sum_hamil = sum_hamil+A*cos(phi)*(gra.Hrf{1,n}) + A*sin(phi)*(gra.Hrf{1,n+length(spinlist)});
        end
        U(:,:,j,k) = expm(-1i*(gra.T/gra.N)*(gra.Hint + sum_hamil));
        X(:,:,j+1,k)=U(:,:,j,k)*X(:,:,j,k);
    end
    
    P(:,:,gra.N,k)=eye(2^gra.nspins);
    for j=fliplr(2:gra.N)
        P(:,:,j-1,k)=P(:,:,j,k)*U(:,:,j,k);
    end
    
    fid(k)= (abs(trace(gra.RHO_targ'*X(:,:,gra.N+1,k)*gra.RHO_init*X(:,:,gra.N+1,k)')));
end


fidelity = fid*gra.rfINHiwt;

