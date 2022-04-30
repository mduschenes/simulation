
function rungrape(ParamsFile,sec)

% Defining some paths
MolPath = [pwd filesep 'MoleculeFiles' filesep ];
InputPath = [pwd filesep 'InputFiles' filesep ];
SavePathDecom = [pwd filesep 'SaveOutputs' filesep 'SaveOutputsDecom' filesep];
SavePath = [pwd filesep 'SaveOutputs' filesep 'SaveOutputsGRAPE' filesep];
ToolsPath = [pwd filesep 'Tools'];
MainFilesPath = [pwd filesep 'MainPrograms' filesep 'MainProgramsGRAPE'];
RFIpath = [pwd filesep 'MainPrograms' filesep 'RFIFiles' filesep];

addpath(ToolsPath);
addpath(MainFilesPath);

% Run the user define input file and Molecule file
run([InputPath ParamsFile]);
run([MolPath GR.MoleculeFile]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load([SavePathDecom 'UEncoding8PC'],'-mat','U');
% ['GR.Utarg = U{' num2str(sec) '};']
% eval(['GR.Utarg = U{' num2str(sec) '};']); 
% filnam1=['Uencoding_' num2str(sec) 'a2us'];
% GR.GRAPEname = filnam1;
% filnam=['Uencoding_' num2str(4) 'a2us'];
% GR.GuessFileName=filnam;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[Ix,Iy,Iz,IHx,IHy,IHz,D] = prodopSparse(Mol.spinNumbers,Mol.spinlist);
for j=1:length(Mol.spinlist)
    IHX{j}=sparse(IHx(:,:,j)); IHY{j}=sparse(IHy(:,:,j));
end

% Finding out number of spins
Mol.nspins = sum(Mol.spinlist);

% Generating free-Evolution Hamiltonian
Hint = genHint(Mol.spinlist,Mol.v,Mol.J,D,Ix,Iy,Iz);

% Some operators for avoiding matrix exponentiation
Had = (1/sqrt(2))*[1 1; 1 -1];
for j=2:Mol.nspins
    Had=kron(Had,(1/sqrt(2))*[1 1; 1 -1]);
end

UH0 = expm(-1i*Hint*GR.del_t/2);
W1 = UH0*Had;
W2 = Had*UH0;
% 3
% Total time of the sequence
GR.T=GR.del_t*GR.N;
% Number of control parameters
GR.m = 2*length(Mol.spinlist); 

% Negative Evolution Unitary
Ud = sparse(expm(1i*GR.initdelay*Hint));

% Unitary for which GRAPE will be optimized
UTopt = Ud*GR.Utarg*Ud;

%%%%%%%%%%%%%%%%%%%%  Checking the disk to prevent overwriting %%%%%%%%%%%%%%%%%%%% 
% Defining the name to be checked as fileName
fileName = [GR.GRAPEname '.mat'];

% Checking if file exist, and owerwrite option, if it does, find a filename_number
% that does not.
if GR.FileOverwrite == 0
     filexist = 0;
     if exist(fileName,'file') == 2
        filexist = 1;
     end
     while filexist > 0
        fileName = [GR.GRAPEname '_' num2str(filexist) '.mat'];
        if exist(fileName,'file') == 2
           filexist = filexist + 1;
        else
           filexist = 0;
        end
     end
end
fileNameTemp = fileName(1:end-4);
fileName = [SavePath fileName];

%%%%%%%%%%%%%%%%%%%%  Penalty Function  %%%%%%%%%%%%%%%%%%%% 
% Max amp for each channel in radians
for j=1:GR.m/2; maxAmp(j) = (2*pi)/4./GR.plength(j); end   
% Convert percentage to number of divisions
DivN = fix(GR.N*(GR.Div/100)); 
% % Generate the penalty curve
x=linspace(-2*pi,2*pi,DivN*2); penaltyCurve = sech(x'); 
penaltyCurveFull=[penaltyCurve(1:DivN); ones(GR.N-2*DivN,1); penaltyCurve(DivN+1:end)];
% % Generate penalty for each control 
PenaltyRange=zeros(GR.N,GR.m/2);
for j=1:GR.m/2
    PenaltyRange(:,j) = 0.95*maxAmp(j)/sqrt(2)*penaltyCurveFull;
end
%%%%%%%%%%%%%%%%%%%% Guess and apply Penalty %%%%%%%%%%%%%%%%%%%% 

% Check if user asked to load a guess or generate a random one
if GR.LoadGuess==0
    u = MakeControls(GR.R,GR.N,GR.m,maxAmp,PenaltyRange);
elseif GR.LoadGuess==1
    load([ SavePath GR.GuessFileName],'-mat','u');
elseif GR.LoadGuess==2
    load([ SavePath GR.GuessFileName],'-mat','u');
    u1=[];
    for j=1:GR.m/2
        uu1 = smooth(u(:,2*j-1),0.15,'loess');
        uu2 = smooth(u(:,2*j),0.15,'loess');
        u1   = [u1 uu1 uu2];
%         Amp=sqrt(u(:,2*j-1).^2+u(:,2*j).^2);
%         Pha=atan2(u(:,2*j),u(:,2*j-1));
%         Amp=smooth(Amp,0.2,'loess');
        %         Pha=smooth(Pha,0.05,'loess');
%         u1=[u1 Amp.*cos(Pha) Amp.*sin(Pha)];
    end
    u=u1;
    clear u1
elseif GR.LoadGuess==3
    load([ SavePath GR.GuessFileName],'-mat','u');
    B=reshape(u,1,length(u)*GR.m);
    u=reshape([B;B],length(u)*2,GR.m);
elseif GR.LoadGuess==4
    load([ SavePath GR.GuessFileName],'-mat','u');
    u1=spline([1:5:500],u(:,1),[1:1:500]);
    u2=spline([1:5:500],u(:,2),[1:1:500]);
    u=[u1' u2'];
    size(u)
else
    fprintf('ErrorMessage')
end

%%%%%%%%%%%%%%%%%%%% Main Program  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Prints the logo saved in the file named logo_grape
logo_grape()

% calculate gradient and fidelity subtraction due to penalizing controls
[GrPenal,PenalFid] = Penalty(u,PenaltyRange,GR.m);
% 4
[Fid,Uf] = CalcFidelity(u,Mol.nspins,GR.N,GR.m,GR.rfi,IHz,W1,W2,UTopt,GR.del_t);
% Fid
% safsd
Fid = Fid-GR.W*PenalFid; % Adjust fidelity with peanlizing controls

counter=0;

% Printing Initial things
fprintf('-------------------------------------------------------------------------------\n')
fprintf('                Target =  %g              Guess Fid = %0.6f         \n',GR.TargFid,Fid)
fprintf('                         SaveFileName = %s         \n',fileNameTemp)
fprintf('-------------------------------------------------------------------------------\n')
fprintf('  Fidelity  |  StepSize  | Max(u)(per)   |  Max(ConjGrad)  |  Iteration  |  Time(s) \n');

tic
oldGrad=zeros(GR.N,GR.m);
oldDirc=zeros(GR.N,GR.m);
ImprvFid = zeros(1,20);
resetflag=0;

alphas = [];

lasttime = cputime;
dspCheck = max((GR.TargFid-Fid)*0.2,1e-4)+Fid;
save(fileName,'GR','Mol','u','Fid')
% sdff
while (Fid <GR.TargFid) && (counter < GR.countermax)
    
    counter=counter+1;
    Grad = CalcGrad(Uf,Mol.nspins,GR.rfi,IHX,IHY,GR.N,GR.m,UTopt,GR.del_t,GrPenal,GR.W);

    [goodDirc,multiFac] = ConjugateGrad(resetflag,counter,Grad,oldGrad,oldDirc,Fid,...
        GR.stepsize,Mol.nspins,GR.N,GR.m,GR.rfi,IHz,W1,W2,UTopt,u,GR.del_t,GR.W,PenaltyRange);
    
    oldGrad=Grad;
    oldstepsize=GR.stepsize;
    oldu=u;
    oldDirc=goodDirc;
    oldFid = Fid;
    
    u = u+(multiFac)*GR.stepsize*goodDirc;
    GR.stepsize=sqrt(multiFac)*GR.stepsize;
    [Fid,Uf] = CalcFidelity(u,Mol.nspins,GR.N,GR.m,GR.rfi,IHz,W1,W2,UTopt,GR.del_t);
    [GrPenal,PenalFid] = Penalty(u,PenaltyRange,GR.m);
    Fid = Fid-GR.W*PenalFid;
    % We screwed up, either stepsize was wrong or conjugate gradients were wrong,
    if Fid<oldFid
        resetflag=1;
        [goodDirc,multiFac] = ConjugateGrad(resetflag,counter,Grad,oldGrad,oldDirc,oldFid,...
            oldstepsize,Mol.nspins,GR.N,GR.m,GR.rfi,IHz,W1,W2,UTopt,u,GR.del_t,GR.W,PenaltyRange);
        u = oldu+(multiFac)*oldstepsize*goodDirc;
        GR.stepsize=sqrt(multiFac)*oldstepsize;
        [Fid,Uf] = CalcFidelity(u,Mol.nspins,GR.N,GR.m,GR.rfi,IHz,W1,W2,UTopt,GR.del_t);
        [GrPenal,PenalFid] = Penalty(u,PenaltyRange,GR.m);
        Fid = Fid-GR.W*PenalFid;
        resetflag=0;
    end
    % Display and save results after every 30 secs or if there is 20% improvement
    if (Fid>dspCheck || (cputime-lasttime)>5) || 1
        alphas(counter) = GR.stepsize;
        dspCheck = max((GR.TargFid-Fid)*0.2,1e-4)+Fid;
        lasttime=cputime;
        fprintf('  %6.6f    %3.2e      %3.2f           %3.2e        %5d          %5.2f \n',Fid,GR.stepsize,max(max(u))/(0.95*maxAmp(1)/sqrt(2))*100,max(max(goodDirc)),counter,toc);
        save(fileName,'GR','Mol','u','Fid')
    end

    % Stopping conditions, if stepsize<MinStepsize
    % OR if fidelity improvement<FidCheck in last 20 iterations
    ImprvFid = [Fid-oldFid ImprvFid(1:19)];
    if GR.stepsize<GR.minstepsize
        fprintf('Stepsize became too small \n');
        return
    end
    if abs(mean(ImprvFid))<GR.FidCheck
        fprintf('Fidelity not improving fast enough \n');
        return
    end
end

save(fileName,'GR','Mol','u','Fid')
toc

% plotGRAPE(fileNameTemp)
% ampwr(fileNameTemp)

fid=fopen('alpha.txt','w');
fprintf(fid,'%0.6e,',alphas);
fclose(fid);
