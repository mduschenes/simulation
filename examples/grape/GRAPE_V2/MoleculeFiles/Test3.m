%%%%%%%%%%% Values for Constructing Internal Hamiltonian
% Example, spinlist=[2 1 2] for total of 5 spins out of which 1-2 and 4-5 
% are homonuclear and (1-2),3,(4-5) are hetronuclear
Mol.spinlist=[1 3];           
% Example, spinNumbers=[1/2 1 3/2] 1-2 have spin=1/2, 3rd have spin=1, 
% 4-5 have spin=3/2   
Mol.spinNumbers=[1 1/2];         

% Chemical shift
ChemicalShifts = [ 100 100 6000 4000];
Offset = [ 0 0];

% J-Coupling values
Mol.J=zeros(sum(Mol.spinlist));
Mol.J(1,2)=150;
Mol.J(1,3)=5;
Mol.J(1,4)=4.35;
Mol.J(2,3)=72;
Mol.J(2,4)=1.5;
Mol.J(3,4)=210;
% Mol.J(1,4)=72.4;
% Mol.J(2,4)=72.4;
% Mol.J(3,4)=72.4;
% Subtract Offset form chemical shifts
O=[];
for j=1:length(Mol.spinlist)
        O=[O repmat(Offset(j),1,Mol.spinlist(j))];
end
Mol.v = -(ChemicalShifts-O);

