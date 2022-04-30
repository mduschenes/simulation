function [H_free_evo] = generate_free_evolH(spinlist,v,J)

% This program calculates the Free Evolution hamiltonian or the Hamiltonian
% in absence of Radiofrequency.
% INPUTS : spinNumbers- an array containing the value of the spin defined in one to one
%          corrospondence with the array in spinlist 
%          spinlist- an array whose lenght corresponds to number of
%          hetronuclear nuclei and each entry to number of homonuclear
%          nuclei
%          v - an array consisting of the offset values of every nuclei
%          J - an array consisting the J coupling 

global gra
[gra.Ix,gra.Iy,gra.Iz,~,~,~,sIHz] = prodop(gra.spinNumbers,gra.spinlist);
Ix=gra.Ix; Iy=gra.Iy; Iz=gra.Iz;


%******************CALCULATION OF OFFSET HAMILTONIAN****************************

Ho=zeros(2^sum(spinlist));
for k=1:length(v)
    H_off = -2*pi*v(k)*Iz(:,:,k);
    Ho=Ho+H_off;
end

%*****************CALCULATION OF COUPLING HAMILTONIAN***************************

Hj=zeros(2^sum(spinlist));
for k=1:length(J)
    for n=1:length(J)
        H_coup = 2*pi*J(k,n)*(Ix(:,:,k)*Ix(:,:,n) + Iy(:,:,k)*Iy(:,:,n) + Iz(:,:,k)*Iz(:,:,n));
%         H_coup = 2*pi*J(k,n)*(Iz(:,:,k)*Iz(:,:,n));
        Hj=Hj+H_coup;
    end
end

H_free_evo=Ho+Hj;