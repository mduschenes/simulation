%%%% SAMPLE INPUT FILE

GR=[];

GR.MoleculeFile='TestMatt';
% Input parameters for GRAPE sequence
GR.N = 1000;                                % Number of Time Steps
GR.del_t = 4e-6;                           % Duration of each Time step 
GR.plength = [20e-6 20e-6 20e-6];                % Length of pi/2 pulse 

% Target unitary
% GR.Utarg = [
%     1 0 0 0 0 0 0 0;
%     0 1 0 0 0 0 0 0;
%     0 0 1 0 0 0 0 0;
%     0 0 0 1 0 0 0 0;
%     0 0 0 0 1 0 0 0;
%     0 0 0 0 0 1 0 0;
%     0 0 0 0 0 0 0 1;
%     0 0 0 0 0 0 1 0;
%     ];
% GR.Utarg = [1 0 0 0;0 1 0 0;0 0 0 1;0 0 1 0];
d_ = 2;
n_ = 4;
GR.Utarg = expm(-1j*rand(d_^n_,d_^n_));

GR.GRAPEname='UCNOT_MATT';

% Penalty
GR.Div = 5;                               % Percentage of initial & final points to penalize
GR.W   =  0.2e-10;                          % Weight of penalty( usually left alone) 
GR.R=2;                            % number of points after which a random value has to be chosen
GR.initdelay = 0; %4e-6;                       %initial and final delays

% Learning rate and fidelities
GR.stepsize = 4e+9;      % 5e9                  % Initial Step Size 
GR.TargFid  = 0.999;                         % Target Fidelity 
GR.countermax = 1000;
GR.minstepsize=1e+6;
GR.FidCheck=1e-7;

% Parameters for RF inhomogenity
load([RFIpath 'rfiMatt.txt']);                 % Name of RFI file
GR.rfi = rfiMatt;

% Load guessflag and if yes, name of guess file
GR.LoadGuess = 0;                    % 0 = No guess, 1 = loadfile, 2= loadfile and smooth the controls
GR.GuessFileName = 'CR7_Step2_Last';

% Flag for overwriting file 0=No
GR.FileOverwrite=1;