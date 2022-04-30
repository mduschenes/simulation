function [goodDirc,maxmult] = ConjugateGrad(resetflag,iter,Grad,oldGrad,oldDirc,FidCurrent,stepsize,nspins,N,m,rfi,IHz,W1,W2,UTopt,u,delT,W,PenaltyRange)

if(iter ~= 1 && ~resetflag)
    diffDerivs = Grad - oldGrad;
    beta = sum(sum(Grad.*diffDerivs))/sum(sum(oldGrad.^2));
else
    beta = 0;
end

%Do a sort of reset.  If we have really lost conjugacy then beta will be
%negative.  If we than reset beta to zero then we start with the
%steepest descent again.
beta = max(beta,0);

%Define the good direction as the linear combination
goodDirc = Grad + beta*oldDirc;

% Finding the best step size
multRange = [0 1 2];
FidTemp(1)=FidCurrent;
for j=2:1:3
    uTemp = u+multRange(j)*stepsize*goodDirc;
    [~,PenalFid] = Penalty(uTemp,PenaltyRange,m);
    FidTemp(j) = CalcFidelity(uTemp,nspins,N,m,rfi,IHz,W1,W2,UTopt,delT)-W*PenalFid;
end

%We have three points to fit a quadratic to.  The matrix to obtain
%the [a b c] coordinates for fitting points 0,1,2 is
h=1;
fitcoeffs=[1/(2*h^2) -(1/h)^2 1/(2*h^2); -(1/h) (1/h) 0; 1 0 0]*FidTemp';
% fitcoeffs = [0.5   -1    0.5; -1.5    2   -0.5; ...
%     1.0000         0         0]*FidTemp';

%If the quadratic is negative this method did not work so just
%go for the maximum value
if(fitcoeffs(1) > 0)
    [~,maxindex] = max(FidTemp);
    maxmult = multRange(maxindex);
    %Otherwise choose the maximum of the quadratic
else
    maxmult = -fitcoeffs(2)/fitcoeffs(1)/2;
end

%Move by at least 0.1 and at most 2;
maxmult = min(max(maxmult,0.1),2);

