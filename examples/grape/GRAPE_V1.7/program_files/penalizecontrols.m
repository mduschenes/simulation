function u =penalizecontrols(u)
global gra

for n=1:gra.m/2
    Aa = sqrt(u(:,n).^2+u(:,n+length(gra.spinlist)).^2);
    [aup,bup]=find(Aa>gra.uprange(:,:,n));
    cup=ones(gra.N,1);
    for j=1:length(aup)
        cup(aup(j),bup(j)) = gra.uprange(aup(j),bup(j),n)/Aa(aup(j),bup(j));
    end
    u(:,n)=cup.*u(:,n);
    u(:,n+length(gra.spinlist))=cup.*u(:,n+length(gra.spinlist));
end