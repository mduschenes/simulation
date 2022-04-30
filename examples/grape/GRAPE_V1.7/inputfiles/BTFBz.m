%% Input File parameters do not change
% clc
global gra

%% Input the parameters for molecule information

% gra.spinlist=[3 2];              %example spinlist=[2 1 2] for total of 5 spins out of which 1-2 and 4-5 are homonuclear and (1-2),3,(4-5) are hetronuclear
% gra.spinNumbers=[1/2 1/2];         %example spinNumbers=[1/2 1 3/2] 1-2 have spin=1/2, 3rd have spin=1 , 4-5 have spin=3/2   

gra.spinlist=[3 2];              %example spinlist=[2 1 2] for total of 5 spins out of which 1-2 and 4-5 are homonuclear and (1-2),3,(4-5) are hetronuclear
gra.spinNumbers=[1/2 1/2];         %example spinNumbers=[1/2 1 3/2] 1-2 have spin=1/2, 3rd have spin=1 , 4-5 have spin=3/2   


%% Offset values
v(1) = 6044; 
v(2) = -3680;
v(3) = -6744;  
v(4) = 50;
v(5) = 28;

% J-Coupling values
J=zeros(sum(gra.spinlist));

% J-Coupling values
D=zeros(sum(gra.spinlist));

J(1,2) = 277;     J(2,4) = 106;
J(1,3) = 116;     J(2,5) = 1270;   
J(1,4) = 54;      J(3,4) = 1532;
J(1,5) = 1556;    J(3,5) = 55;
J(2,3) = -26;     J(4,5) = -7.6;


%% Input parameters for GRAPE sequence
gra.m = 2*length(gra.spinlist);  

gra.N = 2400;                               
gra.del_t = 10e-6;                         

%% Initial guess for the GRAPE sequence

u_max = 8000;            
R=25;  

% u=initial_u(u_max,R);
% % % save guess.mat u
load BTFBzU_lp-25x_24m_hemtest.mat
u = GRinfo.u;
% u = zeros(4800,4);
% for i=1:2400
%     u(2*i-1,:) = u1(i,:);
%     u(2*i,:) = u1(i,:);
% end

%% Target parameters

gra.initdelay = 5e-6;         

% Target Operator
[Ix,Iy,Iz,IHx,IHy,IHz,sIHz] = prodop(gra.spinNumbers,gra.spinlist);     

load BTFBzU_lp-25x_48m4.mat;

gra.Utarg = GRinfo.Utarg;

epsilon=5e+7;                   % Initial Step Size

targ_fide=.99;                 % Target Fidelity 

%% Parameters for GRAPE spequence when converting to bruker format 
% gra.GRAPEname =  'BTFBzqft1';       % Name of the GRAPE structure & Bruker shape file.
% gra.gate = 'BTFBzqft1';               % Name displayed in Bruker Shape File (for convinience).

gra.GRAPEname = ['BTFBzU_lp-25x_24m_hemtest'];
gra.gate = ['BTFBzU_lp-25x_24m_hemtest'];               % Name displayed in Bruker Shape File (for convinience).


%% Parameters for RF inhomogenity (Loading the RF inhomogenity files stored in rfifiles folder)

rfi = [...
    0.95 0.15
    1.0 0.7
    1.05 0.15
    ];

% rfi = [1 1];


gra.rfINHrange = rfi(:,1);
gra.rfINHiwt = rfi(:,2);    


%% PENALTY conditions
% % % div = 10;
plength = [17.1e-6 10.1e-6];

%% Stoping conditions
gra.threshold = 1e-15;
stop_counter  = 100;               % Stop after these many iterations if fid is not increasing more than gra.threshold               
saving_fidel  = 0.75;              % Start saving after this saving_fidel Reached   
iter_no       = 30;                % Save after every iter_no when saving_fidel is reached
gra.totaliterno = 1e6;





