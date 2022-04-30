function [GrPenal,PenalFid] = Penalty(u,PenaltyRange,m)

GrPenal = zeros(length(u),m);
PenalFid=0;
for k=1:m
    umUbound=[]; umLbound=[]; HeavSideU=[]; HeavSideL=[];
    umUbound  = u(:,k)-PenaltyRange(:,ceil(k/2));
    umLbound  = u(:,k)+PenaltyRange(:,ceil(k/2));
    HeavSideU = heaviside(u(:,k)-PenaltyRange(:,ceil(k/2)));
    HeavSideL = heaviside(-u(:,k)-PenaltyRange(:,ceil(k/2)));
    GrPenal(:,k) = 2*umUbound.*HeavSideU+2*umLbound.*HeavSideL;
    PenalFid = PenalFid+sum(2*(umUbound.^2).*HeavSideU+2*(umLbound.^2).*HeavSideL);
end
