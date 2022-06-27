function Grad = CalcGradDecomss12(x,nSpin,nSec,VarPerSec,DelayControl,Utarg,Ix,Iy,Iz,Hint,Uf,nSub,nSpinT,SubS)

Ub=cell(nSub,2*nSec+2);
for ss=1:nSub
    for kk=1:2*nSec+2
        Ub{ss,kk}=Uf{ss,end}*Uf{ss,kk}';
    end
end

for n = 1:nSec
    Ievo = n*VarPerSec;
    Ilast = (n-1)*VarPerSec;

    Ph(n,:)= (x(Ilast+1:Ilast+8));
    An(n,:)= (x(Ilast+8+1:Ilast+2*8));
    ti(n)  = x(Ievo);
end
Ilast=VarPerSec*nSec;
Ph(nSec+1,:)=(x(Ilast+1:Ilast+8));
An(nSec+1,:)=(x(Ilast+8+1:Ilast+2*8));

%%%% Revaluate subs to replace all Hs by 8, then evaluate Ix,Iy,Iz for them
for ss=1:nSub
    for nn = 1:length(SubS{ss}); if SubS{ss}(nn)>=8 && SubS{ss}(nn)<=12; SubS{ss}(nn)=8; end;end
    c = unique(SubS{ss});
    for i = 1:length(c)
        if c(i)==8
            counts{ss}(i) = sum(SubS{ss}==c(i)); % number of times each unique value is repeated
        end
    end
    SubS{ss}=unique(SubS{ss},'stable');
%     SubS{ss}
%     counts{ss}
    nSpin(ss) = length(SubS{ss});
end

EvalGrad = @(x,y,z,a) (2*real(trace(x*y')*z)/2^(2*a));

PhGrad = zeros(nSec+1,8);
AnGrad = zeros(nSec+1,8);
tiGrad = zeros(1,nSec);
AnzGrad= zeros(1,8);
for ss=1:nSub
    trxjpj = trace(Uf{ss,end}'*Utarg{ss});
    for n=1:nSec
        for j = 1:nSpin(ss)
            if SubS{ss}(j)==8
                IZz = zeros(size(Iz{ss}(:,j))); for dd=1:counts{ss}(j); IZz = IZz+Iz{ss}(:,j+dd-1); end
                IXx = zeros(size(Ix{ss}(:,:,j))); for dd=1:counts{ss}(j); IXx = IXx+Ix{ss}(:,:,j+dd-1); end
                IYy = zeros(size(Iy{ss}(:,:,j))); for dd=1:counts{ss}(j); IYy = IYy+Iy{ss}(:,:,j+dd-1); end
                if n==1
                    PhGrad(n,SubS{ss}(j)) = EvalGrad(-1i*(Ub{ss,2*n-1}*(IZz.*Uf{ss,2*n-1})-Uf{ss,end}.*transpose(IZz)),Utarg{ss},trxjpj,nSpin(ss))+PhGrad(n,SubS{ss}(j));
                else
                    PhGrad(n,SubS{ss}(j)) = EvalGrad(-1i*(Ub{ss,2*n-1}*(IZz.*Uf{ss,2*n-1})-Ub{ss,2*n-2}*(IZz.*Uf{ss,2*n-2})),Utarg{ss},trxjpj,nSpin(ss))+PhGrad(n,SubS{ss}(j));
                end
                AnGrad(n,SubS{ss}(j)) = EvalGrad(-1i*Ub{ss,2*n-1}*(cos(Ph(n,SubS{ss}(j)))*IXx+sin(Ph(n,SubS{ss}(j)))*IYy)*Uf{ss,2*n-1},Utarg{ss},trxjpj,nSpin(ss))+AnGrad(n,SubS{ss}(j));
            else
                if n==1
                    PhGrad(n,SubS{ss}(j)) = EvalGrad(-1i*(Ub{ss,2*n-1}*(Iz{ss}(:,j).*Uf{ss,2*n-1})-Uf{ss,end}.*transpose(Iz{ss}(:,j))),Utarg{ss},trxjpj,nSpin(ss))+PhGrad(n,SubS{ss}(j));
                else
                    PhGrad(n,SubS{ss}(j)) = EvalGrad(-1i*(Ub{ss,2*n-1}*(Iz{ss}(:,j).*Uf{ss,2*n-1})-Ub{ss,2*n-2}*(Iz{ss}(:,j).*Uf{ss,2*n-2})),Utarg{ss},trxjpj,nSpin(ss))+PhGrad(n,SubS{ss}(j));
                end
                AnGrad(n,SubS{ss}(j)) = EvalGrad(-1i*Ub{ss,2*n-1}*(cos(Ph(n,SubS{ss}(j)))*Ix{ss}(:,:,j)+sin(Ph(n,SubS{ss}(j)))*Iy{ss}(:,:,j))*Uf{ss,2*n-1},Utarg{ss},trxjpj,nSpin(ss))+AnGrad(n,SubS{ss}(j));
            end
        end
        tiGrad(1,n) = EvalGrad(-1i*Ub{ss,2*n}*(((ti(n)/abs(ti(n)))*DelayControl/pi*Hint{ss}).*Uf{ss,2*n}),Utarg{ss},trxjpj,nSpin(ss))+tiGrad(1,n);
    end

    for j = 1:nSpin(ss)
        if SubS{ss}(j)==8
            IZz = zeros(size(Iz{ss}(:,j))); for dd=1:counts{ss}(j); IZz = IZz+Iz{ss}(:,j+dd-1); end
            IXx = zeros(size(Ix{ss}(:,:,j))); for dd=1:counts{ss}(j); IXx = IXx+Ix{ss}(:,:,j+dd-1); end
            IYy = zeros(size(Iy{ss}(:,:,j))); for dd=1:counts{ss}(j); IYy = IYy+Iy{ss}(:,:,j+dd-1); end
            PhGrad(nSec+1,SubS{ss}(j)) = EvalGrad(-1i*(Ub{ss,end-1}*(IZz.*Uf{ss,end-1})-Ub{ss,end-2}*(IZz.*Uf{ss,end-2})),Utarg{ss},trxjpj,nSpin(ss))+PhGrad(n+1,SubS{ss}(j));
            AnGrad(nSec+1,SubS{ss}(j)) = EvalGrad(-1i*Ub{ss,end-1}*(cos(Ph(nSec+1,SubS{ss}(j)))*IXx+sin(Ph(nSec+1,SubS{ss}(j)))*IYy)*Uf{ss,end-1},Utarg{ss},trxjpj,nSpin(ss))+AnGrad(n+1,SubS{ss}(j));
            AnzGrad(SubS{ss}(j)) = EvalGrad(-1i*(IZz.*Uf{ss,end}),Utarg{ss},trxjpj,nSpin(ss))+AnzGrad(SubS{ss}(j));
        else
            PhGrad(nSec+1,SubS{ss}(j)) = EvalGrad(-1i*(Ub{ss,end-1}*(Iz{ss}(:,j).*Uf{ss,end-1})-Ub{ss,end-2}*(Iz{ss}(:,j).*Uf{ss,end-2})),Utarg{ss},trxjpj,nSpin(ss))+PhGrad(n+1,SubS{ss}(j));
            AnGrad(nSec+1,SubS{ss}(j)) = EvalGrad(-1i*Ub{ss,end-1}*(cos(Ph(nSec+1,SubS{ss}(j)))*Ix{ss}(:,:,j)+sin(Ph(nSec+1,SubS{ss}(j)))*Iy{ss}(:,:,j))*Uf{ss,end-1},Utarg{ss},trxjpj,nSpin(ss))+AnGrad(n+1,SubS{ss}(j));
            AnzGrad(SubS{ss}(j)) = EvalGrad(-1i*(Iz{ss}(:,j).*Uf{ss,end}),Utarg{ss},trxjpj,nSpin(ss))+AnzGrad(SubS{ss}(j));
        end
    end
end
PhGrad=PhGrad/nSub;
AnGrad=AnGrad/nSub;
tiGrad=tiGrad/nSub;
AnzGrad=AnzGrad/nSub;
% sfd
Grad=[];
for n=1:nSec
    Grad=[Grad PhGrad(n,:) AnGrad(n,:) tiGrad(1,n)];
end
Grad =[Grad PhGrad(n+1,:) AnGrad(n+1,:) AnzGrad];
% function Grad = CalcGradDecomss12(x,nSpin,nSec,VarPerSec,DelayControl,Utarg,Ix,Iy,Iz,Hint,Uf,nSub,nSpinT,SubS,V)
% 
% Ub=cell(nSub,2*nSec+2);
% for ss=1:nSub
%     for kk=1:2*nSec+2
%         Ub{ss,kk}=Uf{ss,end}*Uf{ss,kk}';
%     end
% end
% 
% for n = 1:nSec
%     Ievo = n*VarPerSec;
%     Ilast = (n-1)*VarPerSec;
%     
%     Ph(n,:)= (x(Ilast+1:Ilast+8));
%     An(n,:)= (x(Ilast+8+1:Ilast+2*8));
%     ti(n)  = x(Ievo);
% end
% Ilast=VarPerSec*nSec;
% Ph(nSec+1,:)=(x(Ilast+1:Ilast+8));
% An(nSec+1,:)=(x(Ilast+8+1:Ilast+2*8));
% 
% TrueSubS=SubS;
% V=2*pi*V;
% %%%% Revaluate subs to replace all Hs by 8, then evaluate Ix,Iy,Iz for them
% for ss=1:nSub
%     for nn = 1:length(SubS{ss}); if SubS{ss}(nn)>=8 && SubS{ss}(nn)<=12; SubS{ss}(nn)=8; end;end
%     c = unique(SubS{ss});
%     for i = 1:length(c)
%         if c(i)==8
%             counts{ss}(i) = sum(SubS{ss}==c(i)); % number of times each unique value is repeated
%         end
%     end
%     SubS{ss}=unique(SubS{ss},'stable');
%     nSpin(ss) = length(SubS{ss});
%     
% end
% EvalGrad = @(x,y,z,a) (2*real(trace(x*y')*z)/2^(2*a));
% 
% PhGrad = zeros(nSec+1,8);
% AnGrad = zeros(nSec+1,8);
% tiGrad = zeros(1,nSec);
% AnzGrad= zeros(1,8);
% for ss=1:nSub
%     trxjpj = trace(Uf{ss,end}'*Utarg{ss});
%     for n=1:nSec
%         for j = 1:nSpin(ss)
%             if SubS{ss}(j)==8
%                 IZz = zeros(size(Iz{ss}(:,j))); for dd=1:counts{ss}(j); IZz = IZz+Iz{ss}(:,j+dd-1); end
%                 if n==1
%                     IXx = zeros(size(Ix{ss}(:,:,j))); for dd=1:counts{ss}(j); IXx = IXx+Ix{ss}(:,:,j+dd-1); end
%                     IYy = zeros(size(Iy{ss}(:,:,j))); for dd=1:counts{ss}(j); IYy = IYy+Iy{ss}(:,:,j+dd-1); end
%                     PhGrad(n,SubS{ss}(j)) = EvalGrad(-1i*(Ub{ss,2*n-1}*(IZz.*Uf{ss,2*n-1})-Uf{ss,end}.*transpose(IZz)),Utarg{ss},trxjpj,nSpin(ss))+PhGrad(n,SubS{ss}(j));
%                     AnGrad(n,SubS{ss}(j)) = EvalGrad(-1i*Ub{ss,2*n-1}*(cos(Ph(n,SubS{ss}(j)))*IXx+sin(Ph(n,SubS{ss}(j)))*IYy)*Uf{ss,2*n-1},Utarg{ss},trxjpj,nSpin(ss))+AnGrad(n,SubS{ss}(j));
%                 else
%                     IXx = zeros(size(Ix{ss}(:,:,j))); for dd=1:counts{ss}(j); IXx = IXx+(cos(Ph(n,SubS{ss}(j))-V(TrueSubS{ss}(j+dd-1))*sum(abs(ti(1:n-1)))*DelayControl/pi))*Ix{ss}(:,:,j+dd-1); end
%                     IYy = zeros(size(Iy{ss}(:,:,j))); for dd=1:counts{ss}(j); IYy = IYy+(sin(Ph(n,SubS{ss}(j))-V(TrueSubS{ss}(j+dd-1))*sum(abs(ti(1:n-1)))*DelayControl/pi))*Iy{ss}(:,:,j+dd-1); end
%                     PhGrad(n,SubS{ss}(j)) = EvalGrad(-1i*(Ub{ss,2*n-1}*(IZz.*Uf{ss,2*n-1})-Ub{ss,2*n-2}*(IZz.*Uf{ss,2*n-2})),Utarg{ss},trxjpj,nSpin(ss))+PhGrad(n,SubS{ss}(j));
%                     AnGrad(n,SubS{ss}(j)) = EvalGrad(-1i*Ub{ss,2*n-1}*(IXx+IYy)*Uf{ss,2*n-1},Utarg{ss},trxjpj,nSpin(ss))+AnGrad(n,SubS{ss}(j));
%                 end
%             else
%                 if n==1
%                     PhGrad(n,SubS{ss}(j)) = EvalGrad(-1i*(Ub{ss,2*n-1}*(Iz{ss}(:,j).*Uf{ss,2*n-1})-Uf{ss,end}.*transpose(Iz{ss}(:,j))),Utarg{ss},trxjpj,nSpin(ss))+PhGrad(n,SubS{ss}(j));
%                 else
%                     PhGrad(n,SubS{ss}(j)) = EvalGrad(-1i*(Ub{ss,2*n-1}*(Iz{ss}(:,j).*Uf{ss,2*n-1})-Ub{ss,2*n-2}*(Iz{ss}(:,j).*Uf{ss,2*n-2})),Utarg{ss},trxjpj,nSpin(ss))+PhGrad(n,SubS{ss}(j));
%                 end
%                 AnGrad(n,SubS{ss}(j)) = EvalGrad(-1i*Ub{ss,2*n-1}*(cos(Ph(n,SubS{ss}(j)))*Ix{ss}(:,:,j)+sin(Ph(n,SubS{ss}(j)))*Iy{ss}(:,:,j))*Uf{ss,2*n-1},Utarg{ss},trxjpj,nSpin(ss))+AnGrad(n,SubS{ss}(j));
%             end
%         end
%         tiGrad(1,n) = EvalGrad(-1i*Ub{ss,2*n}*(((ti(n)/abs(ti(n)))*DelayControl/pi*Hint{ss}).*Uf{ss,2*n}),Utarg{ss},trxjpj,nSpin(ss)) + tiGrad(1,n);
%         for nn=n+1:nSec
%             for j = 1:nSpin(ss)
%                 if SubS{ss}(j)==8
%                     IZz = zeros(size(Iz{ss}(:,j)));
%                     stime = 0; for k=1:nn-1; if k==n; stime=stime+(ti(n)/abs(ti(n))); else; stime=stime+abs(ti(k)); end; end
%                     for dd=1:counts{ss}(j)
%                         IZz = IZz-V(TrueSubS{ss}(j+dd-1))*stime*DelayControl/pi*Iz{ss}(:,j+dd-1);
%                     end
%                     tiGrad(1,n) = EvalGrad(-1i*(Ub{ss,2*nn-1}*(IZz.*Uf{ss,2*nn-1})-Ub{ss,2*nn-2}*(IZz.*Uf{ss,2*nn-2})),Utarg{ss},trxjpj,nSpin(ss))+tiGrad(1,n);
%                 end
%             end
%         end
%         for j = 1:nSpin(ss)
%             if SubS{ss}(j)==8
%                 IZz = zeros(size(Iz{ss}(:,j)));
%                 stime = 0; for k=1:nSec; if k==n; stime=stime+(ti(n)/abs(ti(n))); else; stime=stime+abs(ti(k)); end; end
%                 for dd=1:counts{ss}(j)
%                     IZz = IZz-V(TrueSubS{ss}(j+dd-1))*stime*DelayControl/pi*Iz{ss}(:,j+dd-1);
%                 end
%                 tiGrad(1,n) = EvalGrad(-1i*(Ub{ss,2*nSec+1}*(IZz.*Uf{ss,2*nSec+1})-Ub{ss,2*nSec}*(IZz.*Uf{ss,2*nSec})),Utarg{ss},trxjpj,nSpin(ss))+tiGrad(1,n);
%                 tiGrad(1,n) = EvalGrad(-1i*(-IZz.*Uf{ss,2*nSec+2}),Utarg{ss},trxjpj,nSpin(ss))+tiGrad(1,n);
%             end
%         end
%     end
%     
%     for j = 1:nSpin(ss)
%         if SubS{ss}(j)==8
%             IZz = zeros(size(Iz{ss}(:,j))); for dd=1:counts{ss}(j); IZz = IZz+Iz{ss}(:,j+dd-1); end
%             PhGrad(nSec+1,SubS{ss}(j)) = EvalGrad(-1i*(Ub{ss,end-1}*(IZz.*Uf{ss,end-1})-Ub{ss,end-2}*(IZz.*Uf{ss,end-2})),Utarg{ss},trxjpj,nSpin(ss))+PhGrad(nSec+1,SubS{ss}(j));
%             IXx = zeros(size(Ix{ss}(:,:,j))); for dd=1:counts{ss}(j); IXx = IXx+(cos(Ph(nSec+1,SubS{ss}(j))-V(TrueSubS{ss}(j+dd-1))*sum(abs(ti(1:nSec)))*DelayControl/pi))*Ix{ss}(:,:,j+dd-1); end
%             IYy = zeros(size(Iy{ss}(:,:,j))); for dd=1:counts{ss}(j); IYy = IYy+(sin(Ph(nSec+1,SubS{ss}(j))-V(TrueSubS{ss}(j+dd-1))*sum(abs(ti(1:nSec)))*DelayControl/pi))*Iy{ss}(:,:,j+dd-1); end
%             AnGrad(nSec+1,SubS{ss}(j)) = EvalGrad(-1i*Ub{ss,end-1}*(IXx+IYy)*Uf{ss,end-1},Utarg{ss},trxjpj,nSpin(ss))+AnGrad(nSec+1,SubS{ss}(j));
%             AnzGrad(SubS{ss}(j)) = EvalGrad(-1i*(IZz.*Uf{ss,end}),Utarg{ss},trxjpj,nSpin(ss))+AnzGrad(SubS{ss}(j));
%         else
%             PhGrad(nSec+1,SubS{ss}(j)) = EvalGrad(-1i*(Ub{ss,end-1}*(Iz{ss}(:,j).*Uf{ss,end-1})-Ub{ss,end-2}*(Iz{ss}(:,j).*Uf{ss,end-2})),Utarg{ss},trxjpj,nSpin(ss))+PhGrad(nSec+1,SubS{ss}(j));
%             AnGrad(nSec+1,SubS{ss}(j)) = EvalGrad(-1i*Ub{ss,end-1}*(cos(Ph(nSec+1,SubS{ss}(j)))*Ix{ss}(:,:,j)+sin(Ph(nSec+1,SubS{ss}(j)))*Iy{ss}(:,:,j))*Uf{ss,end-1},Utarg{ss},trxjpj,nSpin(ss))+AnGrad(nSec+1,SubS{ss}(j));
%             AnzGrad(SubS{ss}(j)) = EvalGrad(-1i*(Iz{ss}(:,j).*Uf{ss,end}),Utarg{ss},trxjpj,nSpin(ss))+AnzGrad(SubS{ss}(j));
%         end
%     end
% end
% asdf
% PhGrad=PhGrad/nSub;
% AnGrad=AnGrad/nSub;
% tiGrad=tiGrad/nSub;
% AnzGrad=AnzGrad/nSub;
% % sfd
% Grad=[];
% for n=1:nSec
%     Grad=[Grad PhGrad(n,:) AnGrad(n,:) tiGrad(1,n)];
% end
% Grad =[Grad PhGrad(n+1,:) AnGrad(n+1,:) AnzGrad];
% asdf