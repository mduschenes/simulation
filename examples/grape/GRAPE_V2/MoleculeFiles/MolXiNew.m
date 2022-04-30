%% Values for Constructing Internal Hamiltonian
Mol.spinlist=[1 3];              %example spinlist=[2 1 2] for total of 5 spins out of which 1-2 and 4-5 are homonuclear and (1-2),3,(4-5) are hetronuclear
Mol.spinNumbers=[1/2 1/2];         %example spinNumbers=[1/2 1 3/2] 1-2 have spin=1/2, 3rd have spin=1 , 4-5 have spin=3/2   

% Chemical shift
ChemicalShifts = [ 15479.706 -33126.235  -42679.73  -56448.522];
Offset = [ 15000 -45000];

% J-Coupling values
Mol.J=zeros(sum(Mol.spinlist));
Mol.J(1,2)=-297.71;
Mol.J(1,3)=-275.59;
Mol.J(1,4)= 39.17 ;
Mol.J(2,3)= 64.745 ;
Mol.J(2,4)= 51.517 ;
Mol.J(3,4)=-128.35 ;

% Subtract chemical shifts from offset
O=[];
for j=1:length(Mol.spinlist)
        O=[O repmat(Offset(j),1,Mol.spinlist(j))];
end
Mol.v = ChemicalShifts-O;

