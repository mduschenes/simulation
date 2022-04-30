%%%%%%%%%%% Values for Constructing Internal Hamiltonian
% Example, spinlist=[2 1 2] for total of 5 spins out of which 1-2 and 4-5 
% are homonuclear and (1-2),3,(4-5) are hetronuclear
Mol.spinlist=[3 1];           
% Example, spinNumbers=[1/2 1 3/2] 1-2 have spin=1/2, 3rd have spin=1, 
% 4-5 have spin=3/2   
Mol.spinNumbers=[1/2 1/2];         

% Chemical shift
ChemicalShifts = [ 30020 8779 6245 2692];
Offset = [ 20696 3000];

% J-Coupling values
Mol.J=zeros(sum(Mol.spinlist));
Mol.J(1,2)=57.58;
Mol.J(1,3)=-2.00;
Mol.J(1,4)=-13.19;
Mol.J(2,3)=32.70;
Mol.J(2,4)=133.6;
Mol.J(3,4)=-6.97;

% Subtract Offset form chemical shifts
O=[];
for j=1:length(Mol.spinlist)
        O=[O repmat(Offset(j),1,Mol.spinlist(j))];
end
Mol.v = -(ChemicalShifts-O);

