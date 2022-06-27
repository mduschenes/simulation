% Author : Hemant Katiyar, 30-June-2020, (hkatiyar@uwaterloo.ca)

% helper function to calculate natural Hamiltonian
% use weak coupling approximation, therefore XX and YY coupling are not
% calculated.

function [Hint, OSL] = genHintSS(spinlist,v,J,SS)
n=length(SS);
SL=spinlist;

SLS(1)=0;
for j=1:length(spinlist)
    SLS(j+1)=sum(SL(1:j));
end
OSL = cell(size(SS));
for j=1:length(SS); OSL{j}=zeros(size(SL)); end

for j=1:n
    for k=1:length(SL)
        for l=1:length(SS{j})
            if SS{j}(l)>SLS(k) && SS{j}(l)<=SLS(k+1)
                OSL{j}(k)=OSL{j}(k)+1;
            else
            end
        end
    end
    [tIx,tIy,tIz,~,~,~,D] = prodopSparse(1/2*ones(size(OSL{j})),OSL{j});
    Ho{j}=zeros(D,1);
    for k=1:sum(OSL{j})
        Ho{j}=Ho{j}+2*pi*v(SS{j}(k))*(tIz(:,k));
    end

    Hj{j}=zeros(D,1);
    for k=1:sum(OSL{j})-1
        for m=k+1:sum(OSL{j})
            Hj{j}=Hj{j}+2*pi*J(SS{j}(k),SS{j}(m))*((tIz(:,k).*tIz(:,m)));
        end
    end
    Hj{j} = diag(Hj{j});
    SpinListSum(1)=0;
    for k=1:length(OSL{j})
        SpinListSum(k+1)=sum(OSL{j}(1:k));
    end

    for l=1:length(OSL{j})
        for k=SpinListSum(l)+1:SpinListSum(l+1)-1
            for m=k+1:SpinListSum(l+1)
                Hj{j}=Hj{j}+2*pi*J(SS{j}(k),SS{j}(m))*(tIx(:,:,k)*tIx(:,:,m)+tIy(:,:,k)*tIy(:,:,m));
            end
        end
    end
    Hint{j}=diag(Ho{j})+Hj{j};
end