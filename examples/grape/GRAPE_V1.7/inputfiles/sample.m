%% Input File parameters do not change
clc
global gra

%% Input the parameters for molecule information

gra.spinlist=[1 1];              %example spinlist=[2 1 2] for total of 5 spins out of which 1-2 and 4-5 are homonuclear and (1-2),3,(4-5) are hetronuclear
gra.spinNumbers=[1/2 1/2];         %example spinNumbers=[1/2 1 3/2] 1-2 have spin=1/2, 3rd have spin=1 , 4-5 have spin=3/2   

%% Offset values
v(1) = 0;
v(2) = 0; 

% J-Coupling values
J=zeros(sum(gra.spinlist));
J(1,2)= 209.13;

%% Input parameters for GRAPE sequence
gra.m = 2*length(gra.spinlist);             % Number of control parameters
gra.N = 50;                               % Number of Time Steps
gra.del_t = 2e-6;                          % Duration of each Time step 

%% Initial guess for the GRAPE sequence

u_max = 8000;               % Initial RF-power
R=10;                       % number of points after which a random value has to be chosen                                   
u=initial_u(u_max,R);

% load guess.mat ;  
% for j=length(abc)+1:length(abc)+10
%     abc(j,:)=abc(end-5,:);
% end
% u=abc;
% u=abc1(1:70,:);
% 
%% Target parameters

gra.initdelay = 5e-6;          %initial and final delay

% Target Operator
[Ix,Iy,Iz,IHx,IHy,IHz,sIHz] = prodop(gra.spinNumbers,gra.spinlist);      % Uncomment If Ix Iy Iz needed for writing the operator
o=[1;0]; l=[0;1]; sx=[0 1; 1 0];
% CNOT12=kron(o*o',eye(2))+kron(l*l',sx);
openCNOT12=kron(l*l',eye(2))+kron(o*o',sx);
had=kron([1 1; 1 -1]/sqrt(2),eye(2));
% x90_1=expm(-1i*pi/2*(Ix(:,:,1)));
x90_2=expm(-1i*pi/2*(Ix(:,:,2)));
% y90_1=expm(-1i*pi/2*(Iy(:,:,1)));
% y90_2=expm(-1i*pi/2*(Iy(:,:,2)));
% x180_1=expm(-1i*pi*(Ix(:,:,1)));

Utarg = openCNOT12;
gra.Utarg = Utarg;

epsilon=5e+9;                   % Initial Step Size
targ_fide=.9999;                % Target Fidelity 

%% Parameters for GRAPE spequence when converting to bruker format 
gra.GRAPEname =  'test_';       % Name of the GRAPE sequence when converted to Bruker format and structure file saves in save_structure folder.
gra.gate = 'openCNOT12';               % Name of the gate just for the convenience displayed in the Bruker Shape File.

%% Parameters for RF inhomogenity (Loading the RF inhomogenity files stored in rfifiles folder)

load rfi8_3pt.txt
rfi=rfi8_3pt;

gra.rfINHrange = rfi(:,1);
gra.rfINHiwt = rfi(:,2);
gra.rfINHrange1 = rfi(:,3);

%% PENALTY conditions
% div = 10;
plength = [10.8e-6  8.85e-6] ;

%% Stoping conditions
gra.threshold = 1e-15;
stop_counter  = 500;                % Stop after these many iterations if fidelity is not increasing more than gra.threshold               
saving_fidel  = 0.90;              % Start saving after this saving_fidel Reached   
iter_no       = 30;                % Save after every iter_no when saving_fidel is reached

