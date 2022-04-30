%% Input File
clc
global gra
%% Input the parameters for molecule information

% No. of spins
gra.spinlist=[1 1];                   %example spinlist=[2 1 2] for total of 5 spins out og which 1-2 and 4-5 are homonuclear and (1-2),3,(4-5) are hetronuclear
spinlist=gra.spinlist;

spinNumbers=[1/2 1/2];              %example spinNumbers=[1/2 1 3/2] 1-2 have spin=1/2, 3rd have spin=1 , 4-5 have spin=3/2   
gra.n_spins = sum(spinlist);



[gra.Ix,gra.Iy,gra.Iz,IHx,IHy,IHz,sIHz] = prodop(spinNumbers,spinlist);
Ix=gra.Ix; Iy=gra.Iy; Iz=gra.Iz;

% Offset values
v(1) = 0;
v(2) = 0;     

% J-Coupling values
J=false(gra.n_spins);
J(1,2)= 219;

% Free-Evolution Hamiltonian
gra.free_evol_H = generate_free_evolH(spinlist,v,J);

% RF Hamiltonian H_k's in a cell form, e.g. rf_H= { H_1 , H_2 , H_3 , H_4}
gra.rf_H = generate_rfH(spinlist);

%% Input parameters for GRAPE sequence

% Number of Time Steps
gra.N = 100;    

% Duration of each Time step 
del_t = 5e-6;

% Number of control parameters
gra.m = length(gra.rf_H);  

% Total time of the sequence
gra.T=del_t*gra.N;      

%% Initial guess for the GRAPE sequence

% Initial RF-power
u_max = 40000;

% number of points after which a random value has to be chossen
R=20;                                   

u=initial_u(u_max,R);
guess=u;

%% Target parameters

% target state operator
gra.rho_target = Ix(:,:,1);

% Step Size
epsilon=5e+7;

% Target Fidelity 
targ_fide=.999;

%% Parameters for GRAPE spequence when converting to bruker format 
gra.GRAPEname = 'CHCL3_cnot';
gra.gate = 'CNOT';


%% Stop if Fidelity is not increasing by
gra.threshold = 1e-5;

%% Parameters for RF inhomogenity
% load rfiQXI_H3pt.txt
% rfi=rfiQXI_H3pt;
rfi = [...
   8.0000000e-01   8.5733191e-02
   1.0000000e+00   7.4008489e-01
   1.2000000e+00   1.7418192e-01
];

gra.Range = rfi(:,1);
gra.rfiwt = rfi(:,2);





