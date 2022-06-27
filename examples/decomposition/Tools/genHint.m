% Author : Hemant Katiyar, 30-June-2020, (hkatiyar@uwaterloo.ca)

% helper function to calculate natural Hamiltonian
% use weak coupling approximation, therefore XX and YY coupling are not
% calculated.

function Hint = genHint(spinlist,v,J,D,Ix,Iy,Iz)

% Generating Zeeman Hamiltonian
% We are generating $Ho = \sum_i 2\pi v_i I_z^i$

Ho=zeros(D,1);
for k=1:length(v)
    Ho=Ho+2*pi*v(k)*diag(Iz(:,k));
end

% Generating J-coupling Hamiltonian
% We are generating $Hj = \sum_{ij}^{j>i} 2\pi J (I_z^i I_z^j)$

Hj=zeros(D,1);
for k=1:sum(spinlist)-1
    for n=k+1:sum(spinlist)
        Hj=Hj+2*pi*J(k,n)*(diag(Iz(:,k).*Iz(:,n)));
    end
end

SpinListSum(1)=0;
for j=1:length(spinlist)
    SpinListSum(j+1)=sum(spinlist(1:j));
end

for l=1:length(spinlist)
    for k=SpinListSum(l)+1:SpinListSum(l+1)-1
        for n=k+1:SpinListSum(l+1)
            Hj=Hj+2*pi*J(k,n)*(Ix(:,:,k)*Ix(:,:,n)+Iy(:,:,k)*Iy(:,:,n));
        end
    end
end


%% Total Internal Hamiltonian
Hint=Ho+Hj;
