% Author : Hemant Katiyar, 30-June-2020, (hkatiyar@uwaterloo.ca)

% Function to help decompose complex unitaries into single qubit gates
% and free evolutions


function Udecompose2ss(ParFile)

% Definining Some Paths
MolPath = [pwd filesep 'MoleculeFiles' filesep ];
InputPath = [pwd filesep 'InputFiles' filesep ];
SavePath = [pwd filesep 'SaveOutputs' filesep 'SaveOutputsDecom' filesep];
ExtrasPath = [pwd filesep 'Extras' filesep];
ToolsPath = [pwd filesep 'Tools'];
MainFilesPath = [pwd filesep 'MainPrograms' filesep 'MainProgramsDecom'];
addpath(ToolsPath);
addpath(MainFilesPath);

% Run the user define input file and Molecule file
run([InputPath ParFile]);
run([MolPath Par.MoleculeFile]);


Mol.nSpinTotal=sum(Mol.spinlist);
nSub = length(Par.subsystem);
% Create Hint(free of chemical shift) and HintFull(includes chemical shift)
vtemp=zeros(size(Mol.v));
[Mol.Hint, OSL] = genHintWeakSS(Mol.spinlist,vtemp,Mol.J,Par.subsystem);
% Mol.HintFull = genHintSS(Mol.spinlist,Mol.v,Mol.J,Par.subsystem);
% Mol.Hint = genHintWeak(Mol.spinlist,vtemp,Mol.J,D,Iz);
% Mol.HintFull = genHint(Mol.spinlist,Mol.v,Mol.J,D,Ix,Iy,Iz);

for ss = 1:length(Par.subsystem)
    % Number of spins
    Mol.nSpin(ss) = sum(OSL{ss});
    
    % Generate Paulis
    [Ix{ss},Iy{ss},Iz{ss}] = prodopSparse(1/2*ones(size(OSL{ss})),OSL{ss});
    
    % Hadamard gates acting on all spins
    Had = (1/sqrt(2))*[1 1; 1 -1];
    for j=1:Mol.nSpin(ss)-1
        Had = kron(Had,(1/sqrt(2))*[1 1; 1 -1]);
    end
    Had1{ss}=Had;
end

% Initializing some variables
Par.VarPerSec = Mol.nSpinTotal*2 + 1;
Par.TotVar = Par.VarPerSec*Par.nSec + Mol.nSpinTotal*3;

% Either create initial guess or load a guess depending upon input from user
if Par.GuessFlag==0
    x = InitialGuess(Par.TotVar,Par.nSec,Par.VarPerSec,Par.DelayControl,Par.MaxIniDelay);
else
    load([SavePath Par.GuessFileName],'-mat','x');
%     x= x+(rand(size(x))-0.5)/10;
%     x = [rand(1,15) x];
%     size(x)
end

% Calculate total delay time for guess
DelayTime=0;
for j=1:Par.nSec
    DelayTime=DelayTime+ abs(x(j*Par.VarPerSec))*Par.DelayControl/pi;
end

% Save data in savepath
fileName=[SavePath Par.SaveFileName '.mat'];


% % Function Definition
% CalcFid = @(x,y,z) abs(trace(x*y'))/2^z;


%%%%%%%%%%%%%%%%%%%%%%%%%%%% MAIN PROGRAM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
stepsize = Par.init_stepsize;

[Fid,Uf,Fids] = CalcFidDecomss(x,Mol.nSpin,Par.nSec,Par.VarPerSec,Par.DelayControl,Iz,Had1,Mol.Hint,Par.Utarg,nSub,Mol.nSpinTotal,Par.subsystem);

oldgrad = zeros(1,Par.TotVar);
olddirc = zeros(1,Par.TotVar);

counter=1;
fprintf('  Fidelity  |  StepSize  |  Max(ConjGrad)  |  Iteration  |  FreeEvo(ms)  |  Time(s) \n');
fprintf('  %6.6f    %6.5f        %6.5f        %5d            %5.4f        %5.2f \n',Fid,stepsize,max(olddirc),counter,DelayTime*1e+3,toc);
dspCheck = max((Par.TargFid-Fid)*0.2,1e-4)+Fid;
save(fileName,'Par','Mol','x','Fid')
% Conjugate gradient reset flag, if its 1 we lose conjugacy
resetflag=0;
% guesscount=0;

while Fid<Par.TargFid
    
    Grad = CalcGradDecomss(x,Mol.nSpin,Par.nSec,Par.VarPerSec,Par.DelayControl,Par.Utarg,Ix,Iy,Iz,Mol.Hint,Uf,nSub,Mol.nSpinTotal,Par.subsystem);
%     Grad
%     sdf
    [gooddirc,multiFac] = ConjugateGradDecomss(resetflag,Mol.nSpin,counter,Grad,oldgrad,olddirc,Fid,x,stepsize,Iz,Had1,Mol.Hint,Par.nSec,Par.VarPerSec,Par.DelayControl,Par.Utarg,nSub,Mol.nSpinTotal,Par.subsystem);
    
    resetflag=0;
    
    oldgrad = Grad;
    oldstepsize = stepsize;
    oldx = x;
    oldFid = Fid;
    olddirc = gooddirc;
    
    x = x+multiFac*stepsize*gooddirc;
    stepsize = sqrt(multiFac)*stepsize;
    [Fid,Uf,Fids] = CalcFidDecomss(x,Mol.nSpin,Par.nSec,Par.VarPerSec,Par.DelayControl,Iz,Had1,Mol.Hint,Par.Utarg,nSub,Mol.nSpinTotal,Par.subsystem);
    
    % We screwed up, either fit was not accurate or conjugate gradient are bad
    % Do a more fine linsesearch
    if Fid<oldFid
        resetflag=1;
        mulFineRange = linspace(0,4,201);
        FidTemp = zeros(1,length(mulFineRange));
        for j=1:length(mulFineRange)
            xtemp = oldx+mulFineRange(j)*oldstepsize*Grad;
            FidTemp(j)= CalcFidDecomss(xtemp,Mol.nSpin,Par.nSec,Par.VarPerSec,Par.DelayControl,Iz,Had1,Mol.Hint,Par.Utarg,nSub,Mol.nSpinTotal,Par.subsystem);
        end
        [~,indx] = max(FidTemp);
        multiFacTemp = mulFineRange(indx);
        multiFac = max(0.01,multiFacTemp);
        x = oldx+multiFac*stepsize*Grad;   
        stepsize=oldstepsize*sqrt(multiFac);
        [Fid,Uf,Fids] = CalcFidDecomss(x,Mol.nSpin,Par.nSec,Par.VarPerSec,Par.DelayControl,Iz,Had1,Mol.Hint,Par.Utarg,nSub,Mol.nSpinTotal,Par.subsystem);
    end
    
    counter=counter+1;
    
    % Displaying and saving current variables
    if (Fid>dspCheck || mod(counter,50)==0)
        save(fileName,'Par','Mol','x','Fid')
%         FixPhases(Par.SaveFileName)
        DelayTime=0;
        for j=1:Par.nSec
            dd(j)=abs(x(j*Par.VarPerSec))*Par.DelayControl/pi;
            DelayTime=DelayTime+abs(x(j*Par.VarPerSec))*Par.DelayControl/pi;
        end
        
        fprintf('  %6.6f    %6.5f        %6.5f        %5d            %5.4f          %5.2f \n',Fid,stepsize,max(olddirc),counter,DelayTime*1e+3,toc);
        dspCheck = max((Par.TargFid-Fid)*0.2,1e-4)+Fid;
    end
    
    % Check if the gradients become too small or there is not much improvement
    % in the fidelity. Depending on KeepTrying, we either stop and print out
    % the reason for stopping or start with a new guess
    if max(gooddirc)<1e-5 || abs(Fid-oldFid)<1e-9
        Fids
        if Par.KeepTrying==0 %If user asked not to try new guesses
            if max(gooddirc)<1e-5
                fprintf('Try new guess, small gradients \n');
            elseif abs(Fid-oldFid)<1e-9
                fprintf('Try new guess, small fidelity improvement \n');
            end
            return
        else %If user asked to keep on trying the guess
            if max(gooddirc)<1e-5
                fprintf('Try new guess, small gradients \n');
            elseif abs(Fid-oldFid)<1e-9
                fprintf('Try new guess, small fidelity improvement \n');
            end
            fprintf('Trying a new guess \n');
            Fids
            x = InitialGuess(Par.TotVar,Par.nSec,Par.VarPerSec,Par.DelayControl,Par.MaxIniDelay);
            DelayTime=0;
            for j=1:Par.nSec
                DelayTime=DelayTime+abs(x(j*Par.VarPerSec))*Par.DelayControl/pi;
            end
            stepsize=Par.init_stepsize;
            [Fid,Uf,Fids] = CalcFidDecomss(x,Mol.nSpin,Par.nSec,Par.VarPerSec,Par.DelayControl,Iz,Had1,Mol.Hint,Par.Utarg,nSub,Mol.nSpinTotal,Par.subsystem);

            oldgrad = zeros(1,Par.TotVar);
            olddirc = zeros(1,Par.TotVar);
            counter=1;
            fprintf('  Fidelity  |  StepSize  |  Max(ConjGrad)  |  Iteration  |  FreeEvo(ms)  |  Time(s) \n');
            fprintf('  %6.6f    %6.5f        %6.5f        %5d            %5.4f        %5.2f \n',Fid,stepsize,max(olddirc),counter,DelayTime*1000,toc);
            dspCheck = max((Par.TargFid-Fid)*0.2,1e-4)+Fid;
        end
    end
end %Big while loop
Fids
save(fileName,'Par','Mol','x','Fid')
% FixPhases(Par.SaveFileName)

%%%%%%%%%%%%%%%%%%%%%%%%%% MAIN PROGRAM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add penalty on time
% write correcting phases program auto ---- Done
% test with other stepsize function ---- Better results
% test new conjugate method ----- Bad results
% Add overwrite Prevention
