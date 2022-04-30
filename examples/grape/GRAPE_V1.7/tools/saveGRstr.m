function GRinfo  = saveGRstr(gra,str_name)
if nargin<1
    disp('  ')
    disp('      saveGRstr(GRinfo,''structure_name'')')
    disp('      saves all the data output  to save_strucuture folder so that the tools can work ')
    disp('      at some later time by loading the mat file in the workspace')
    disp('  ')
    disp('         structure_name - Name of the mat file e.g. ''my_struct''')
    disp('                      {default: whatever the name for the GRAPE pulse is given}')
    disp('  ')
    disp('      SIMPLEST INPUT : saveGRstr(GRinfo)')
    disp('  ')
    disp('  ')
    disp('   It is unnecessary to provide all the inputs.  Empty inputs indicate')
    disp('   indicate default selections.')
    disp('  ')
    disp('   (Hemant Katiyar, 2012)')
    return
end

if (nargin<2 || isempty(str_name)); str_name = gra.struct_name; end

GRinfo.GRAPEname=gra.GRAPEname;
GRinfo.struct_name=gra.struct_name;
GRinfo.gate=gra.gate;
GRinfo.Date=date;

GRinfo.spinlist=gra.spinlist;
GRinfo.nspins=gra.nspins;
GRinfo.spinNumbers=gra.spinNumbers;

GRinfo.m=gra.m;
GRinfo.N=gra.N;
GRinfo.del_t=gra.del_t;
GRinfo.T=gra.T;

GRinfo.Chem_shift=gra.v;
GRinfo.Couplings=gra.J;
GRinfo.Hint=gra.Hint;
GRinfo.Hrf=gra.Hrf;

GRinfo.initdelay=gra.initdelay;
GRinfo.Utarg=gra.Utarg;
GRinfo.Usim=(gra.Ud)'*gra.Usim*(gra.Ud)';

GRinfo.rfINHrange=gra.rfINHrange;
GRinfo.rfINHiwt=gra.rfINHiwt;

GRinfo.threshold=gra.threshold;
GRinfo.ep=gra.ep;
GRinfo.mfa=gra.mfa;

GRinfo.u=gra.u;
GRinfo.IDEALfidelity=gra.IDEALfidelity;
GRinfo.RFfidelity=gra.RFfidelity;

fileName = [str_name '.mat'];
fileAddr = ['save_structure' filesep fileName];
save( fileAddr , 'GRinfo')