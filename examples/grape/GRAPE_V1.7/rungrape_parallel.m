function GRinfo = rungrape_parallel(InputName,div,askplot,FileOpt)
%% %%%%%%%%%%%%%%%%%%%%%%% INFORMATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin<1
   disp(' ')
   disp('                                 GRAPE 1.3                   ')
   disp('                                                             ')
   disp('        Construction of GRAPE(gradient ascent pulse engineering)  ')
   disp('               Hemant Katiyar*, T. S. Mahesh* and Koteshwar Rao^        ')
   disp('      *Department of Physics and NMR, Indian Institute of Science Education and Research, Pune. ')
   disp('             ^Department of NMR, Indian Institute of Science       ')
   disp('                          (September 2012)                ')
   disp('                                                             ')
   disp('                                                             ')
   disp('                                                             ')
   disp('  GRinfo = rungrape(''InputFile'',''FileOpt'')')
   disp(' ')
   disp(' ')
   disp('  GRinfo     --> A structured output containing some useful informations about')
   disp('                the GRAPE Pulse constructed which is saved in ''save_structure'' ')
   disp('                folder with the name supplied in input file under GRAPEname')
   disp(' ')
   disp(' ')
   disp(' ''InputFile'' --> A script (.m) file containing input parameters; see examples.')
   disp(' ')
   disp('           div --> percentage of initial and final amplitude to be bounded{default:10}')
   disp(' ')
   disp('  ''askplot'' --> ''y'' or ''n'' for for seeing the penalty function{default:''n''}')
   disp(' ')
   disp(' ''FileOpt'' --> Overwrite or save as a new file?')
   disp('                ''n'' [for new] / ''o'' [for overwrite] (default ''n''). ')
   disp('                The output filename is ''GRAPEnameK.mat'', where K is automatically')
   disp('                selected unless overwriting is opted and GRAPEname is provided in the inputfile') 
   disp(' ')
   disp('  Simplest input: GRinfo = rungrape(''InputFile'')');
   disp('  Empty inputs indicate default selections. ')
   disp(' ')
   disp(' ')
   disp('  Disclaimer: GRAPE is a free program with no claims.');
   disp('              Bugs can be reported to hemantkatiyar7@gmail.com.');
   disp('              Suggestions and comments are also most welcome.');
   disp(' ')
   disp(' ')
   disp(' ')
   return
end

%% %%%%%%%%%%%%%%%%%%%%%%%% DECLARE GLOBAL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
clc
clear global
ti=cputime();
global gra

%% %%%%%%%%%%%%%%%%%%%%%%%% DEFAULT INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
if (nargin < 2 | isempty(div)); div = 5    ; end;
if (nargin < 3 | isempty(askplot)); askplot = 'n'; end;
if (nargin < 4 | isempty(FileOpt)); FileOpt = 'n'; end;

%% %%%%%%%%%%%%%%%%%%%%%%%% Add Paths  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path(path,[pwd filesep 'program_files'])
path(path,[pwd filesep 'tools' ])
path(path,[pwd filesep 'Bruker_Output'])
path(path,[pwd filesep 'inputfiles'])
path(path,[pwd filesep 'rfifiles'])
path(path,[pwd filesep 'save_structure'])

%% %%%%%%%%%%%%%%%%%%%%% INPUT PARAMETER INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eval(InputName);

gra.v=v;
gra.J=J;

gra.nspins = sum(gra.spinlist);
[gra.Ix,gra.Iy,gra.Iz,~,~,~,sIHz] = prodop(gra.spinNumbers,gra.spinlist);

% Free-Evolution Hamiltonian
gra.Hint = generate_free_evolH(gra.spinlist,v,J);

% RF Hamiltonian H_k's in a cell form, e.g. rf_H= { H_1 , H_2 , H_3 , H_4}
gra.Hrf = generate_rfH(gra.spinlist);


gra.T=gra.del_t*gra.N;                                    % Total time of the sequence
% gra.m = length(gra.rf_H);                                 % Number of control parameters
gra.Ud = expm(1i*gra.Hint*gra.initdelay);         %Negative evolution Unitary
gra.U_target = gra.Ud*gra.Utarg*gra.Ud;                  %Unitary for which GRAPE will be optimized


%% %%%%%%%%%%% CHECKING TO PREVENT OVERWRITING FILE %%%%%%%%%%%%%
fileName = [gra.GRAPEname '.mat'];
gra.struct_name = gra.GRAPEname;
if FileOpt == 'n'
     filexist = 0;
     if exist(fileName,'file') == 2
        filexist = 1;
     end
     while filexist > 0; 
        fileName = [gra.GRAPEname num2str(filexist) '.mat'];
        gra.struct_name = [gra.GRAPEname num2str(filexist)];                  % Name of the mat file that will be saved having all the information
        if exist(fileName,'file') == 2
           filexist = filexist + 1;
        else
           filexist = 0;
        end
     end
end
gra.GRAPEname = gra.struct_name;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%% PENALTY %%%%%%%%%%%%%%%%%%%%%%%%%%
[gra.uprange,exitflag]= penalty(div,plength,askplot);
if exitflag
    return
end

u=penalizecontrols(u);


%% %%%%%%%%%%%%%%%%%%%%%%%% MAIN PROGRAM %%%%%%%%%%%%%%%%%%%%%%%%

logo_grape()
% fprintf(' Target Fidelity: %g\n',targ_fide)

[Fid P X] = calculate_robustfidelity_parallel(u);
gra.F(1)=Fid;
counter=0;
n=0;

fprintf('-------------------------------------------------------\n')
fprintf('       Current Fidelity           Iteration            \n')
fprintf('       Target =  %g                    \n',targ_fide)
fprintf('       Guess Fid = %0.6f \n',Fid)
fprintf('-------------------------------------------------------\n')


while (Fid <targ_fide)
    counter=counter+1;
    [epsilon u skip_calc_fidelity U X]=maximise_robustfidelity_parallel(P,X,u,epsilon,counter,Fid);
    gra.ep(counter)=epsilon;
    if skip_calc_fidelity==1
        [Fid P X] = calculate_robustfidelity_exp_not_needed(U,X);
    else
        [Fid P X] = calculate_robustfidelity_parallel(u);
    end
    if(mod(counter,10)==0)
        fprintf('  %20.6f  %14g \n',Fid,counter)
    end
    gra.F(counter+1)=Fid;
    if(gra.F(counter)>saving_fidel)
        if(mod(n,iter_no)==0)
            gra.u=u;
            [gra.IDEALfidelity X1] = calculate_fidelity(u);
            gra.Usim=X1(:,:,gra.N+1);
            gra.RFfidelity = gra.F(length(gra.F));
            GRinfo = saveGRstr(gra);
            fprintf('\t\t DATA SAVED IN save_structure/%s\n',gra.struct_name)
        end
        n=n+1;
    end
    if (counter>stop_counter && gra.F(counter)-gra.F(counter-stop_counter)<gra.threshold)     
        al=1;
        break
    end
    al=2;
end


%% %%%%%%%%%%%%%%%%%%%%%%%%  Saving after the program has finished %%%%%%%%%%%%%%%%%%%%%%%%

gra.u=u;
[gra.IDEALfidelity X1] = calculate_fidelity(u);
gra.Usim=X1(:,:,gra.N+1);
gra.RFfidelity = gra.F(length(gra.F));
GRinfo =saveGRstr(gra);
plotGRAPE(GRinfo)

gra.reason = {'Fidelity Not Improving','Target Fidelity Reached'};

%% %%%%%%%%%%%%%%%%%%%%%%%%  Print Result  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
e = cputime-ti; %e=e/60;
logo_grape()
fprintf('-------------------------------------------------------\n')
fprintf('Termination Reason : %s \n',gra.reason{al})
fprintf(' Target Fidelity  : %g\n Fidelity Reached : %g\n Fideliey of Usim : %g\n Time Taken\t\t  : %g seconds\n Iterations\t\t  : %g\n',targ_fide,gra.RFfidelity,gra.IDEALfidelity,e,counter)
fprintf('-------------------------------------------------------\n')
fprintf('DATA SAVED IN save_structure/%s\n',GRinfo.struct_name)

