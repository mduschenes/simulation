function [fidelity X] = calculate_fidelity(u)

global gra

X(:,:,1)=eye(2^gra.nspins);
U=zeros(2^gra.nspins,2^gra.nspins,gra.N);
for j=1:gra.N
    su=zeros(2^gra.nspins);
    for k=1:gra.m
        su=su + u(j,k)*gra.Hrf{1,k};
    end
    U(:,:,j) = expm(-1i*(gra.T/gra.N)*(gra.Hint + su));
    X(:,:,j+1)=U(:,:,j)*X(:,:,j);
end


fidelity = (abs(trace(gra.U_target'*X(:,:,gra.N+1)))/2^(gra.nspins))^2;

