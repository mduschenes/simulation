function [fidelity P X] = calculate_robustfidelity_exp_not_needed(U,X)
global gra


P=zeros(2^gra.nspins,2^gra.nspins,gra.N,length(gra.rfINHrange));

for k=1:length(gra.rfINHrange)
    
    P(:,:,gra.N,k)=eye(2^gra.nspins);
    for j=fliplr(2:gra.N)
        P(:,:,j-1,k)=P(:,:,j,k)*U(:,:,j,k);
    end
    fid(k)= (abs(trace(gra.U_target'*X(:,:,gra.N+1,k)))/2^(gra.nspins))^2;

end

fidelity = fid*gra.rfINHiwt;

