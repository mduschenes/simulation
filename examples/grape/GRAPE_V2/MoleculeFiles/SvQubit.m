%%%%%%%%%%% Values for Constructing Internal Hamiltonian
% Example, spinlist=[2 1 2] for total of 5 spins out of which 1-2 and 4-5 
% are homonuclear and (1-2),3,(4-5) are hetronuclear
Mol.spinlist=[7];           
% Example, spinNumbers=[1/2 1 3/2] 1-2 have spin=1/2, 3rd have spin=1, 
% 4-5 have spin=3/2   
Mol.spinNumbers=[1/2];         

% Chemical shift
% ChemicalShifts = [ 30020 8779 6245 10333 15745 34381 11928];
ChemicalShifts = [30022.4451 8779.9831 6244.6332 10332.2755 15744.9908 34384.6712 11927.6233];
Offset = [20696];

% J-Coupling values
Mol.J=zeros(sum(Mol.spinlist));
Mol.J(1,2)=57.58;
Mol.J(1,3)=-2.00;
Mol.J(1,4)=0.02;
Mol.J(1,5)=1.43;
Mol.J(1,6)=5.54;
Mol.J(1,7)=-1.43;
Mol.J(2,3)=32.70;
Mol.J(2,4)=.30;
Mol.J(2,5)=2.62;
Mol.J(2,6)=-1.66;
Mol.J(2,7)=37.58;
Mol.J(3,4)=0;
Mol.J(3,5)=-1.10;
Mol.J(3,6)=0;
Mol.J(3,7)=.94;
Mol.J(4,5)=33.16;
Mol.J(4,6)=-3.53;
Mol.J(4,7)=29.02;
Mol.J(5,6)=33.16;
Mol.J(5,7)=21.75;
Mol.J(6,7)=34.57;

% Subtract Offset form chemical shifts
O=[];
for j=1:length(Mol.spinlist)
        O=[O repmat(Offset(j),1,Mol.spinlist(j))];
end
Mol.v = -(ChemicalShifts-O);

