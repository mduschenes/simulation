function Fid = CheckFidelity(InputFile)

load(InputFile,'-mat','GR','u','Mol');

[Ix,Iy,Iz,IHx,IHy,~,D] = prodopSparse(Mol.spinNumbers,Mol.spinlist);
Hint =genHint(Mol.spinlist,Mol.v,Mol.J,D,Ix,Iy,Iz);

% Negative Evolution Unitary
Ud = expm(1i*GR.initdelay*Hint);
% Unitary for which GRAPE will be optimized
UTopt = Ud*GR.Utarg*Ud;

Fid=0;
for l=1:size(GR.rfi,1)
    U = eye(2^Mol.nspins);
    for j=1:GR.N
        Hrf=zeros(2^Mol.nspins);
        for k=1:GR.m/2
            A = GR.rfi(l,1)*sqrt(u(j,2*k-1)^2+u(j,2*k)^2);
            P = atan2(u(j,2*k),u(j,2*k-1));
            Hrf = Hrf + A*(cos(P)*IHx(:,:,k)+sin(P)*IHy(:,:,k));
        end
        U = expm(-1i*GR.del_t*(Hint+Hrf))*U;
    end
    Fid=Fid+GR.rfi(l,2)*(abs(trace(UTopt'*U))/2^(Mol.nspins));
end

