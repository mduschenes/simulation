%% Input File parameters do not change
clc
global gra

%% Input the parameters for molecule information

gra.spinlist=[3];              %example spinlist=[2 1 2] for total of 5 spins out of which 1-2 and 4-5 are homonuclear and (1-2),3,(4-5) are hetronuclear
gra.spinNumbers=[1/2];         %example spinNumbers=[1/2 1 3/2] 1-2 have spin=1/2, 3rd have spin=1 , 4-5 have spin=3/2   

%% Offset values
v(1) = 15765.68;
v(2) = 0; 
v(3) = -4325;  

% J-Coupling values
J=zeros(sum(gra.spinlist));
J(1,2) = 54.02;
J(2,3) = 34.8; 
J(1,3) = 1.19;

%% Input parameters for GRAPE sequence
gra.m = 2*length(gra.spinlist);             % Number of control parameters
gra.N = 120;                               % Number of Time Steps
gra.del_t = 2e-6;                          % Duration of each Time step 

%% Initial guess for the GRAPE sequence

u_max = 8000;               % Initial RF-power
R=10;                       % number of points after which a random value has to be chosen                                   
u=initial_u(u_max,R);
% load guess.mat ;  
% for j=length(abc)+1:length(abc)+10
%     abc(j,:)=abc(end-15,:);
% end
% u =abc;

%% Target parameters

gra.initdelay = 5e-6;          %initial and final delay

% Target Operator
[Ix,Iy,Iz,IHx,IHy,IHz,sIHz] = prodop(gra.spinNumbers,gra.spinlist);      % Uncomment If Ix Iy Iz needed for writing the operator
gx=2*Ix(:,:,1); gz=-4*Iy(:,:,1)*Iy(:,:,3);
gx1=4*Ix(:,:,1)*Iz(:,:,3); gz1=-4*Ix(:,:,1)*Ix(:,:,3);

be=-pi/4; ne=-3*pi/4;
% be=3*pi/4; ne=pi/4;
% be=pi/4; ne=3*pi/4;
% be=-3*pi/4; ne=-pi/4;

% c_0=kron(kron(eye(2),[1 0; 0 0]),eye(2));
% c_1=kron(kron(eye(2),[0 0; 0 1]),eye(2));

% A=gx; cA=c_0 + c_1*A;
% B=cos(be)*gx1 + sin(be)*gz1; cB=c_0 + c_1*B;
% C=gz; cC=c_0 + c_1*C;
% D=cos(ne)*gx1 + sin(ne)*gz1; cD=c_0 + c_1*D;

x60_all=expm(-1i*pi/3*(Ix(:,:,1)+Ix(:,:,3)+Ix(:,:,2)));
x45_3=expm(-1i*pi/4*(Ix(:,:,3)));

gra.Utarg = x45_3;

epsilon=5e+8;                   % Initial Step Size
targ_fide=.9999;                 % Target Fidelity 

%% Parameters for GRAPE spequence when converting to bruker format 
gra.GRAPEname =  'Cont_x45_3';       % Name of the GRAPE sequence when converted to Bruker format and structure file saves in save_structure folder.
gra.gate = 'Contx45_3';               % Name of the gate just for the convenience displayed in the Bruker Shape File.

%% Parameters for RF inhomogenity (Loading the RF inhomogenity files stored in rfifiles folder)

load rfi8_5pt.txt
rfi=rfi8_5pt;
     
     
gra.rfINHrange = rfi(:,1);
gra.rfINHiwt = rfi(:,2);    

%% PENALTY conditions
% % % div = 10;
plength = 17.6e-6;

%% Stoping conditions
gra.threshold = 1e-15;
stop_counter  = 100;               % Stop after these many iterations if fidelity is not increasing more than gra.threshold               
saving_fidel  = 0.9;              % Start saving after this saving_fidel Reached   
iter_no       = 30;                % Save after every iter_no when saving_fidel is reached

