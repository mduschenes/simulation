function [fidelity X] = calculate_fidelity_parallel(u)

global gra
Number_of_steps = gra.N;
t=(gra.T/gra.N);
Hinternal=gra.Hint;

X(:,:,1)=eye(2^gra.nspins);
U=zeros(2^gra.nspins,2^gra.nspins,gra.N);
su_dummy=zeros(2^gra.nspins,2^gra.nspins,gra.N);

for j=1:gra.N
    su=zeros(2^gra.nspins);
    for k=1:gra.m
        su=su + u(j,k)*gra.Hrf{1,k};
    end
	su_dummy(:,:,j)=su;
end

parfor j=1:Number_of_steps	
    U(:,:,j) = expm(-1i*t*(Hinternal + su_dummy(:,:,j)));
end

for j=1:gra.N
    X(:,:,j+1)=U(:,:,j)*X(:,:,j);
end


fidelity = (abs(trace(gra.U_target'*X(:,:,gra.N+1)))/2^(gra.nspins))^2;

