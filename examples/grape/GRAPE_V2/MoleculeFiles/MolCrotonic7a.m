%%%%%%%%%%% Values for Constructing Internal Hamiltonian
% Example, spinlist=[2 1 2] for total of 5 spins out of which 1-2 and 4-5 
% are homonuclear and (1-2),3,(4-5) are hetronuclear
Mol.spinlist=[4 5];           
% Example, spinNumbers=[1/2 1 3/2] 1-2 have spin=1/2, 3rd have spin=1, 
% 4-5 have spin=3/2   
Mol.spinNumbers=[1/2 1/2];         

% Chemical shift
% ChemicalShifts = [ 29341.874 21591.773 25461.146 2990.283 4863.558 4089.4945 1312.814 1312.814 1312.814];
ChemicalShifts = [29342.473 21591.773 25462.2998 2991.1047 4863.558 4089.4945 1312.814 1312.814 1312.814]; %OCT18-21
Offset = [ 16165.30 3086.5];

% J-Coupling values
Mol.J=zeros(sum(Mol.spinlist));
Mol.J(1,2)=72.4;
Mol.J(1,3)=1.4;
Mol.J(1,4)=7.1;
Mol.J(1,5)=6.5;
Mol.J(1,6)=3.3;
Mol.J(1,7)=-0.9;
Mol.J(1,8)=-0.9;
Mol.J(1,9)=-0.9;

Mol.J(2,3)=69.7;
Mol.J(2,4)=1.6;
Mol.J(2,5)=-1.8;
Mol.J(2,6)=162.9;
Mol.J(2,7)=6.6;
Mol.J(2,8)=6.6;
Mol.J(2,9)=6.6;

Mol.J(3,4)=41.6;
Mol.J(3,5)=156;
Mol.J(3,6)=-0.7;
Mol.J(3,7)=-7.1;
Mol.J(3,8)=-7.1;
Mol.J(3,9)=-7.1;

Mol.J(4,5)=3.8;
Mol.J(4,6)=6.2;
Mol.J(4,7)=127.5;
Mol.J(4,8)=127.5;
Mol.J(4,9)=127.5;

Mol.J(5,6)=15.5;
Mol.J(5,7)=6.9;
Mol.J(5,8)=6.9;
Mol.J(5,9)=6.9;

Mol.J(6,7)=-1.7;
Mol.J(6,8)=-1.7;
Mol.J(6,9)=-1.7;

% Subtract Offset form chemical shifts
O=[];
for j=1:length(Mol.spinlist)
        O=[O repmat(Offset(j),1,Mol.spinlist(j))];
end
Mol.v = -(ChemicalShifts-O);

