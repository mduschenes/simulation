%%%%%%%%%%% Values for Constructing Internal Hamiltonian
% Example, spinlist=[2 1 2] for total of 5 spins out of which 1-2 and 4-5 
% are homonuclear and (1-2),3,(4-5) are hetronuclear
Mol.spinlist=[4];           
% Example, spinNumbers=[1/2 1 3/2] 1-2 have spin=1/2, 3rd have spin=1, 
% 4-5 have spin=3/2   
Mol.spinNumbers=[1/2];         

% Chemical shift
% ChemicalShifts = [ 29347.581375 21591.55125 25463.8862 2991.573375];
% ChemicalShifts = [ 29343.19223 21591.54758 25463.29400 2991.61978];
% ChemicalShifts = [ 29342.7977 21591.7256 25463.0354 2991.5595];
ChemicalShifts = [29342.473 21591.773 25462.2998 2991.1047]; %OCT18-21
Offset = [ 16165.30];

% J-Coupling values
Mol.J=zeros(sum(Mol.spinlist));
Mol.J(1,2)=2*36.1349;
Mol.J(1,3)=2*0.5789;
Mol.J(1,4)=2*3.5188;
Mol.J(2,3)=2*34.8400;
Mol.J(2,4)=2*0.7207;
Mol.J(3,4)=2*20.8276;
% Mol.J(1,2)=2*36.1335;
% Mol.J(1,3)=2*0.5825;
% Mol.J(1,4)=2*3.5175;
% Mol.J(2,3)=2*34.8330;
% Mol.J(2,4)=2*0.7235;
% Mol.J(3,4)=2*20.8260;



% Subtract Offset form chemical shifts
O=[];
for j=1:length(Mol.spinlist)
        O=[O repmat(Offset(j),1,Mol.spinlist(j))];
end
Mol.v = ChemicalShifts-O;
Mol.v = -Mol.v;



% 3.459259607747635
% 3.587830080367394
% 6.001651654535328
