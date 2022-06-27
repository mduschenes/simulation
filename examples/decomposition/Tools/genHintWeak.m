% Author : Hemant Katiyar, 30-June-2020, (hkatiyar@uwaterloo.ca)

% helper function to calculate natural Hamiltonian
% use weak coupling approximation, therefore XX and YY coupling are not
% calculated.
%
% Output Hint is diagonal, therfore we store it as a column vector

function Hint = genHintWeak(spinlist,v,J,D,Iz)

% Generating Zeeman Hamiltonian
% We are generating $Ho = -\sum_i 2\pi v_i I_z^i$

Ho=zeros(D,1);
for k=1:length(v)
    Ho=Ho+2*pi*v(k)*(Iz(:,k));
end

% Generating J-coupling Hamiltonian
% We are generating $Hj = \sum_{ij}^{j>i} 2\pi J (I_z^i I_z^j)$

Hj=zeros(D,1);
for k=1:sum(spinlist)-1
    for n=k+1:sum(spinlist)
        Hj=Hj+2*pi*J(k,n)*((Iz(:,k).*Iz(:,n)));
    end
end

%% Total Internal Hamiltonian
Hint=Ho+Hj;
