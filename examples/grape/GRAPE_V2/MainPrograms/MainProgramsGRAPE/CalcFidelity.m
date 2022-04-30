% Works for multiple channel
function [Fid,Uforward] = CalcFidelity(u,nspins,N,m,rfi,IHz,W1,W2,UTopt,delT)

Uforward = cell(N,size(rfi,1));

AIZ=zeros(2^nspins,N);PIZ=zeros(2^nspins,N);
doubleu= u.^2;
for k=1:m/2
    AAA = repmat(IHz(:,k),1,N);
    PIZ = PIZ-(1i*atan2(u(:,2*k),u(:,2*k-1))').*AAA;
    AIZ = AIZ-((1i*delT)*sqrt(sum(doubleu(:,[2*k-1 2*k]),2))').*AAA;
end
for j=1:N  
    PPMAT = exp(PIZ(:,j)).*exp(PIZ(:,j))';
    for l=1:size(rfi,1)
        if j==1
            Uforward{j,l}  = (PPMAT.*(W1*(exp(rfi(l,1)*AIZ(:,j)).*W2)));
        else
            Uforward{j,l}  = (PPMAT.*(W1*(exp(rfi(l,1)*AIZ(:,j)).*W2)))*Uforward{j-1,l};
        end
    end
end
Fid=0;
for l=1:size(rfi,1)
    Fid=Fid+rfi(l,2)*(abs(trace(UTopt'*Uforward{end,l}))/2^(nspins));
end
