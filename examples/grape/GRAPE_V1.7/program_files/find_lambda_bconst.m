function lambda = find_lambda_bconst(iter)
global gra

gra.gr(:,:,iter)=gra.af;
bet=0;


if (iter == 1)
    gra.lambda(:,:,1)=gra.gr(:,:,1);
else
    if ((gra.F(iter)-gra.F(iter-1))<1e-8)
        bet=0;
    else
        bet=sum(sum(gra.gr(:,:,iter).*(gra.gr(:,:,iter)-gra.gr(:,:,iter-1))))/sum(sum(gra.gr(:,:,iter-1).^2));
        bet=max(0,bet);
    end
    gra.lambda(:,:,iter)=gra.gr(:,:,iter) + bet*gra.lambda(:,:,iter-1);
end

lambda=gra.lambda(:,:,iter);
