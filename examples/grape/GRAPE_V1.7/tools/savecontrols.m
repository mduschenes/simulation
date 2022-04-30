function savecontrols(GRinfo,name)
if nargin<1
    disp('  ')
    disp('      savecontrols(GRinfo,name)')
    disp('      Saves the controls parameters(u) in controls folder')
    disp('  ')
    disp('           name - Name of the mat file to be saved as')
    disp('  ')
    disp('      SIMPLEST INPUT : savecontrols(GRinfo,Name)')
    disp('')
    disp('  ')
    disp('  ')
    disp('   (Hemant Katiyar, 2012)')
    return
end


global gra
gra=GRinfo; 
cntrl = gra.u;

fileName = [name '.mat'];
fileAddr = ['controls' filesep fileName];
save( fileAddr , 'cntrl')