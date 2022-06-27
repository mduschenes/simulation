Par=[];

% Name of molecule file located in MoleculeFiles folder
Par.MoleculeFile='MolCrotonic7';

%----------------------------------------------------------------------
%---------- Finding controlled gate -----------------------------------
%---------- control on qubit 4 and target simultaneously on 5 and 7. --

Par.subsystem = {[4 5 7],[3 4 5 7], [2 3 6],[1 2]};   % each array select the subsystems
O = [1;0]; l = [0;1];X = [0 1; 1 0]; I = eye(2); II = eye(2^2); III = eye(2^3); IIII = eye(2^4);
ll = l*l'; OO = O*O';
Par.Utarg{1} = kron(I,kron(I,OO))+kron(X,kron(X,ll));
Par.Utarg{2} = kron(I,kron(I,kron(I,OO)))+kron(I,kron(X,kron(X,ll)));
Par.Utarg{3} = III;
Par.Utarg{4} = II;
Par.SaveFileName = 'CR7_CNOT457_ss';
Par.GuessFlag = 0;   % If GuessFlag=0, then no guess is used, if its 1 load the GuessFileName
Par.GuessFileName = '-';
Par.nSec = 6;    % number of sections in decomposition
Par.KeepTrying=0; % If algorithm stops prematurely, should we keep trying? if yes, then 1
%---------------------------------------------------------------------

% Max value of sum of all delay times for the guess (in ms)
Par.MaxIniDelay = 8;
% Parmeter to control the delay times(usually left alone)
Par.DelayControl = 0.001;
% Initial step size(usually left alone)
Par.init_stepsize=.5;
% Target fidelity of decomposition needed
Par.TargFid = 0.999;

