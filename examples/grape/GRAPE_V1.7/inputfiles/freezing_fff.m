%% Input File parameters do not change
clc
global gra

%% Input the parameters for molecule information

gra.spinlist=[3];              %example spinlist=[2 1 2] for total of 5 spins out of which 1-2 and 4-5 are homonuclear and (1-2),3,(4-5) are hetronuclear
gra.spinNumbers=[1/2];         %example spinNumbers=[1/2 1 3/2] 1-2 have spin=1/2, 3rd have spin=1 , 4-5 have spin=3/2   

%% Offset values
v(1) = 11860.8;
v(2) = 0; 
v(3) = -17379.09;  

% J-Coupling values
J=zeros(sum(gra.spinlist));
J(1,2) = 69.9;
J(2,3) = -128.25; 
J(1,3) = 47.4;

%% Input parameters for GRAPE sequence
gra.m = 2*length(gra.spinlist);             % Number of control parameters
gra.N = 1000;                               % Number of Time Steps
gra.del_t = 10e-6;                          % Duration of each Time step 

%% Initial guess for the GRAPE sequence

u_max = 8000;               % Initial RF-power
R=100;                       % number of points after which a random value has to be chosen                                   
u=initial_u(u_max,R);

%% Target parameters

gra.initdelay = 5e-6;          %initial and final delay

% Target Operator
[Ix,Iy,Iz,IHx,IHy,IHz,sIHz] = prodop(gra.spinNumbers,gra.spinlist);      % Uncomment If Ix Iy Iz needed for writing the operator

% x90_all=expm(-1i*pi/2*(Ix(:,:,1)+Ix(:,:,3)+Ix(:,:,2)));

gra.Utarg = expm(-i*(pi/4)*Ix(:,:,2));

epsilon=5e+7;                   % Initial Step Size

targ_fide=.999;                 % Target Fidelity 

%% Parameters for GRAPE spequence when converting to bruker format 
gra.GRAPEname =  'Context_45x2_';       % Name of the GRAPE sequence when converted to Bruker format and structure file saves in save_structure folder.
gra.gate = 'U84';               % Name of the gate just for the convenience displayed in the Bruker Shape File.

%% Parameters for RF inhomogenity (Loading the RF inhomogenity files stored in rfifiles folder)

load rfi8_5pt.txt
rfi=rfi8_5pt;
     
     
gra.rfINHrange = rfi(:,1);
gra.rfINHiwt = rfi(:,2);    

%% PENALTY conditions
plength = 17.6e-6;

%% Stoping conditions
gra.threshold = 1e-15;
stop_counter  = 100;               % Stop after these many iterations if fidelity is not increasing more than gra.threshold               
saving_fidel  = 0.95;              % Start saving after this saving_fidel Reached   
iter_no       = 60;                % Save after every iter_no when saving_fidel is reached

