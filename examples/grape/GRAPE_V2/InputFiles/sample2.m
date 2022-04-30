%%%% SAMPLE INPUT FILE

GR=[];

 GR.MoleculeFile='Test2';
%GR.MoleculeFile='MolCrotonic7';
% GR.MoleculeFile='TwQubit';
% GR.MoleculeFile='SvQubit';
% Input parameters for GRAPE sequence
GR.N = 500;                                % Number of Time Steps
GR.del_t = 2e-6;                           % Duration of each Time step 

% Initial guess for the GRAPE sequence
GR.plength = [20e-6 20e-6];                % Length of pi/2 pulse 
GR.R=2;                            % number of points after which a random value has to be chosen

% Load guessflag and if yes, name of guess file
GR.LoadGuess = 0;                    % 0 = No guess, 1 = loadfile, 2= loadfile and smooth the controls
GR.GuessFileName = 'CR7_Step2_Last';

% Flag for overwriting file 0=No
GR.FileOverwrite=1;

% Penalty
GR.Div = 5;                               % Percentage of initial & final points to penalize
GR.W   =  0.2e-10;                          % Weight of penalty( usually left alone) 

% Target parameters 
GR.initdelay = 4e-6;                       %initial and final delays

% Some operators and running MoleculeFile
% [Ix,~,~,IHx,~,~,~] = prodopSparse([1/2 1/2],[4 5]);
[Ix,~,~,IHx,~,~,~] = prodopSparse([1/2],[2]);
%GR.GRAPEname='CR9_90yALL_ss';
%GR.GRAPEname='CR9_90xM_ss';
%GR.GRAPEname='CR9_RaXbar_ss'; %.7
%GR.GRAPEname='CR9_XRb_ss';
%GR.GRAPEname='CR9_90xC4_ss';
%GR.GRAPEname='CR9_180xC4M_ss';
%GR.GRAPEname='CR9_90yALLbutM_ss';
%GR.GRAPEname='CR9_U90yC4_ss';
%GR.GRAPEname='CR9_U90myC4_ss';
%GR.GRAPEname='CR9_90yALL_ss';
%GR.GRAPEname='CR9_90yM_ss';
%GR.GRAPEname='CR9_90myM_ss';

% GR.Utarg = expm(-1i*(acos(1/2)*Ix(:,:,2)+acos(1/4)*Ix(:,:,3)+acos(1/8)*Ix(:,:,4))); GR.GRAPEname =  'mPCrush_C2C3C4x';
% GR.Utarg = expm(-1i*(acos(1/2)*Ix(:,:,2)+acos(1/4)*Ix(:,:,3)+pi/2*Ix(:,:,4))); GR.GRAPEname='Ucrushr';
% GR.Utarg = expm(-1i*(acos(1/2)*Ix(:,:,2)+acos(1/4)*Ix(:,:,3))); GR.GRAPEname='UcrushR';

% GR.Utarg = expm(-1i*(pi/4*Ix(:,:,1))); GR.GRAPEname =  'U45x1';
% GR.Utarg = expm(-1i*(pi/4*Ix(:,:,2))); GR.GRAPEname =  'U45x2';
% GR.Utarg = expm(-1i*(pi/4*Ix(:,:,3))); GR.GRAPEname =  'U45x3';

% GR.Utarg = expm(-1i*pi*(Ix(:,:,1)+Ix(:,:,2))); GR.GRAPEname = 'U180x12';
% GR.Utarg = expm(-1i*pi*(Ix(:,:,2)+Ix(:,:,3))); GR.GRAPEname = 'U180x23';
% GR.Utarg = expm(-1i*pi*(Ix(:,:,3)+Ix(:,:,4))); GR.GRAPEname = 'U180x34';

% GR.Utarg = expm(-1i*(pi/2*(Ix(:,:,1)+Ix(:,:,2)+Ix(:,:,3)+Ix(:,:,4)))); GR.GRAPEname='U90xALLtest';
GR.Utarg = expm(-1i*(pi/2*(Ix(:,:,1)+Ix(:,:,2)))); GR.GRAPEname='U90xALLtest';
% GR.Utarg = expm(-1i*(pi/2*(Ix(:,:,1)))); GR.GRAPEname='U90x1a';
% cnot = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0];
% cnot = kron(cnot,eye(2^2)); 
% swap = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1];
% swap = kron(kron(eye(2),swap),eye(2));
% GR.GRAPEname='Cnot';
% GR.Utarg = cnot;
% GR.Utarg = expm(-1i*(pi/2*(Ix(:,:,7)))); GR.GRAPEname='U90xM_Cr7';

% GR.Utarg = expm(-1i*(pi/2*Ix(:,:,7))); GR.GRAPEname='U90x7_svR3';
% GR.Utarg = expm(-1i*(pi/2*(Ix(:,:,1)+Ix(:,:,2)+Ix(:,:,3)+Ix(:,:,4)+Ix(:,:,5)+Ix(:,:,6)))); GR.GRAPEname='U90x1to6_svR4';
% GR.Utarg = expm(-1i*(pi/2*IHx)); GR.GRAPEname='U90xAll_svR4';
% GR.Utarg = expm(-1i*(pi/2*Ix(:,:,2))); GR.GRAPEname='U90x2_svR3'; 
% AA=Ix(:,:,1)+Ix(:,:,2)+Ix(:,:,3)+Ix(:,:,4)+Ix(:,:,5)+Ix(:,:,6);
% GR.Utarg = expm(-1i*(pi/2*AA)); GR.GRAPEname='U90x1to6_sv';
% GR.Utarg = expm(-1i*(pi/2*IHx)); GR.GRAPEname='U90xAll_sva';
% GR.Utarg = expm(-1i*(pi/2*(Ix(:,:,3)))); GR.GRAPEname='U90x3S';
% GR.Utarg = expm(-1i*(pi/2*(Ix(:,:,4)))); GR.GRAPEname='U90x4S';
% GR.Utarg = expm(-1i*(pi/2*(Ix(:,:,7)+Ix(:,:,8)+Ix(:,:,9)))); GR.GRAPEname='U90xM_CR9_Full';
% load ('G:\My Drive\CurrentMatlab\Compiler_strong_Editing\UTT.mat')
% GR.Utarg = UTT; GR.GRAPEname='CR7_Step2_Last';

% The=[5.87829 5.78584 5.70181 5.62611 5.55832 5.49779 5.44379 5.39558 5.35248 5.31384 5.27912 5.24781 5.21949 5.19379 5.17039 5.14902 5.12944 5.11145 5.09488 5.07956];
% Indx=6; GR.Utarg = expm(-1i*(The(Indx)*(Ix(:,:,1)))); GR.GRAPEname=['Xtheta' num2str(Indx)];

GR.stepsize = 5e+9;                        % Initial Step Size 
GR.TargFid  = 0.999;                         % Target Fidelity 

% Parameters for RF inhomogenity
load([RFIpath 'rfi9_5pt']);                 % Name of RFI file
GR.rfi = rfi9_5pt;
% load([RFIpath 'rfi97_3pt']);                 % Name of RFI file
% GR.rfi = rfi97_3pt;
% GR.rfi = [1 1];

% Stopping conditions
GR.minstepsize=1e+7;
GR.FidCheck=1e-7;
