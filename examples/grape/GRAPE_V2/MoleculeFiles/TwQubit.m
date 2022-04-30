%%%%%%%%%%% Values for Constructing Internal Hamiltonian
% Example, spinlist=[2 1 2] for total of 5 spins out of which 1-2 and 4-5 
% are homonuclear and (1-2),3,(4-5) are hetronuclear
Mol.spinlist=[7 5];           
% Example, spinNumbers=[1/2 1 3/2] 1-2 have spin=1/2, 3rd have spin=1, 
% 4-5 have spin=3/2   
Mol.spinNumbers=[1/2 1/2];         

% Chemical shift
ChemicalShifts = [ 30020 8779 6245 10333 15745 34381 11928 3310 2468 2158 2692 3649];
Offset = [ 20696 3000];

% J-Coupling values
Mol.J=zeros(sum(Mol.spinlist));
Mol.J(1,2)=57.58;
Mol.J(1,3)=-2.00;
Mol.J(1,4)=0;
Mol.J(1,5)=1.25;
Mol.J(1,6)=5.54;
Mol.J(1,7)=-1.25;
Mol.J(1,8)=0;
Mol.J(1,9)=4.41;
Mol.J(1,10)=1.81;
Mol.J(1,11)=-13.19;
Mol.J(1,12)=7.87;
Mol.J(2,3)=32.70;
Mol.J(2,4)=.90;
Mol.J(2,5)=2.62;
Mol.J(2,6)=-1.66;
Mol.J(2,7)=37.58;
Mol.J(2,8)=0;
Mol.J(2,9)=1.56;
Mol.J(2,10)=3.71;
Mol.J(2,11)=133.6;
Mol.J(2,12)=-8.35;
Mol.J(3,4)=0;
Mol.J(3,5)=-1.11;
Mol.J(3,6)=0;
Mol.J(3,7)=.94;
Mol.J(3,8)=2.36;
Mol.J(3,9)=146.6;
Mol.J(3,10)=146.6;
Mol.J(3,11)=-6.97;
Mol.J(3,12)=3.35;
Mol.J(4,5)=33.16;
Mol.J(4,6)=-3.53;
Mol.J(4,7)=29.02;
Mol.J(4,8)=166.6;
Mol.J(4,9)=2.37;
Mol.J(4,10)=2.37;
Mol.J(4,11)=6.23;
Mol.J(4,12)=8.13;
Mol.J(5,6)=33.16;
Mol.J(5,7)=21.75;
Mol.J(5,8)=4.06;
Mol.J(5,9)=0;
Mol.J(5,10)=0;
Mol.J(5,11)=0;
Mol.J(5,12)=2.36;
Mol.J(6,7)=34.57;
Mol.J(6,8)=5.39;
Mol.J(6,9)=0;
Mol.J(6,10)=0;
Mol.J(6,11)=5.39;
Mol.J(6,12)=8.52;
Mol.J(7,8)=8.61;
Mol.J(7,9)=0;
Mol.J(7,10)=0;
Mol.J(7,11)=3.78;
Mol.J(7,12)=148.5;
Mol.J(8,9)=0;
Mol.J(8,10)=0.18;
Mol.J(8,11)=-0.68;
Mol.J(8,12)=8.46;
Mol.J(9,10)=-12.41;
Mol.J(9,11)=1.28;
Mol.J(9,12)=-1.06;
Mol.J(10,11)=6;
Mol.J(10,12)=-0.36;
Mol.J(11,12)=1.30;

% Subtract Offset form chemical shifts
O=[];
for j=1:length(Mol.spinlist)
        O=[O repmat(Offset(j),1,Mol.spinlist(j))];
end
Mol.v = -(ChemicalShifts-O);

