%
% clear
% 
% 
% n=4;
% [Ix]=prodopSparse(1/2,n);
% Ix1=gpuArray(Ix);
% 
% tic
% Iy=zeros(2^n,2^n,n);
% for j=1:n
%     Iy(:,:,j) = Ix(:,:,j)*Ix(:,:,j);
% end
% toc
% 
% 
% 
% tic
% 
% Iy1=pagefun(@mtimes,Ix1,Ix1);
% % wait(gpuDevice)
% % Iy2=gather(Iy1);
% toc
% 
% %%
% clear
% B=rand(2^7);
% A=gpuArray(B);
% 
% tic
% expm(B);
% toc
% tic
% expm(A);
% % wait(gpuDevice);
% toc
%%

% clear
% n=7;
% [Ix,Iy,Iz,~,~,~,D]=prodopSparse(1/2,n);
% spinlist=n;
% v=rand(n,1);
% J=rand(n,n);
% 
% tic
% % Ho=zeros(D,1);
% % for k=1:length(v)
% %     Ho=Ho+2*pi*v(k)*diag(Iz(:,k));
% % end
% Hj=zeros(D,1);
% for k=1:sum(spinlist)-1
%     for n=k+1:sum(spinlist)
%         Hj=Hj+2*pi*J(k,n)*(diag(Iz(:,k).*Iz(:,n))+Ix(:,:,k)*Ix(:,:,n)+Iy(:,:,k)*Iy(:,:,n));
%     end
% end
% toc
% 
% for jj=1:n
%     Ix1{jj}=Ix(:,:,jj);
%     Iy1{jj}=Iy(:,:,jj);
%     Iz1{jj}=Iz(:,jj);
% end
% tic
% % % Ho1=zeros(D,1);
% % % parfor k=1:length(v)
% % %     Ho1=Ho1+2*pi*v(k)*diag(Iz(:,k));
% % % end
% J1=nonzeros(transpose(triu(J,1)));
% AA=[];
% for k=1:sum(spinlist)-1
%     for n=k+1:sum(spinlist)
%         AA = [AA ; k n];
%     end
% end
% Hj1=zeros(D,1);
% for k=1:length(J1)
%     Hj1=Hj1+2*pi*J1(k)*(diag(Iz1{AA(k,1)}.*Iz1{AA(k,2)})+Ix1{AA(k,1)}*Ix1{AA(k,2)}+Iy1{AA(k,1)}*Iy1{AA(k,2)});
% end
% toc

%% its better to define spare then sparse in a cell then full (can only help in Hamiltonian)
clear
n=7;
[Ix,Iy,Iz,~,~,~,D]=prodopSparse(1/2,n);

tic
for j=1:45
   A1=diag(Iz(:,1).*Iz(:,2))+Ix(:,:,1)*Ix(:,:,2)+Iy(:,:,1)*Iy(:,:,2);
end
toc

Ix1=sparse(Ix(:,:,1));
Iy1=sparse(Iy(:,:,1));
Ix2=sparse(Ix(:,:,2));
Iy2=sparse(Iy(:,:,2));
Iz1=sparse(diag(Iz(:,1)));
Iz2=sparse(diag(Iz(:,2)));
tic
for j=1:45
   B1=Iz1*Iz2+Ix1*Ix2+Iy1*Iy2;
end
toc



Ixx{1}=sparse(Ix(:,:,1));
Ixx{2}=sparse(Ix(:,:,2));
Iyy{1}=sparse(Iy(:,:,1));
Iyy{2}=sparse(Iy(:,:,2));
Izz{1}=sparse(diag(Iz(:,1)));
Izz{2}=sparse(diag(Iz(:,2)));
tic
for j=1:45
   B2=Izz{1}*Izz{2}+Ixx{1}*Ixx{2}+Iyy{1}*Iyy{2};
end
toc

%%
clear
n=7;
[Ix,Iy,Iz,IHx,~,~,D]=prodopSparse(1/2,n);
N=500;
for j=1:N
    for l=1:3
        Uf{j,l}=rand(2^n,2^n);
    end
end

tic
Ub=cell(N,3);
for l=1:3
    for j=1:N
        Ub{j,l} = Uf{end,l}*Uf{j,l}';
    end
end
toc

tic
for l=1:3
    for j=1:N
        Ufmat(:,:,l,j) = Uf{j,l}';
    end
end
Uff=gpuArray(Ufmat);
Ufend = gpuArray(Ufmat(:,:,:,end));

Ubb = pagefun(@mtimes,Ufend,Ufmat);
wait(gpuDevice)
toc
