%%% Values for Constructing Internal Hamiltonian
Mol.spinlist=[1 3];              %example spinlist=[2 1 2] for total of 5 spins out of which 1-2 and 4-5 are homonuclear and (1-2),3,(4-5) are hetronuclear
Mol.spinNumbers=[1/2 1/2];         %example spinNumbers=[1/2 1 3/2] 1-2 have spin=1/2, 3rd have spin=1 , 4-5 have spin=3/2   

% Chemical shift
ChemicalShifts = [ 15480 -33090 -42647 -56427];
Offset = [ 15500 -45000];

% J-Coupling values
Mol.J=zeros(sum(Mol.spinlist));
Mol.J(1,2)=-148.8550;
Mol.J(1,3)=-138.3355;
Mol.J(1,4)=19.5805;
Mol.J(2,3)=32.419 ;
Mol.J(2,4)=25.792;
Mol.J(3,4)=-64.8755;
Mol.J=2*Mol.J;

% Subtract chemical shifts from offset
O=[];
for j=1:length(Mol.spinlist)
        O=[O repmat(Offset(j),1,Mol.spinlist(j))];
end
Mol.v = ChemicalShifts-O;
Mol.v = -Mol.v;

