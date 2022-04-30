function plotGRAPE(GRinfo)

if nargin<1
    disp('  ')
    disp('      plotGRAPE(GRinfo)')
    disp('      plots variation of controls, each figure corresponds to a channel')
    disp('  ')
    disp('  ')
    disp('      SIMPLEST INPUT : plotGRAPE(GRinfo)')
    disp('')
    disp('  ')
    disp('  ')
    disp('   (Hemant Katiyar, 2012)')
    return
end
global gra
gra=GRinfo;
u=gra.u;


for j=1:gra.m/2
    figure
    plot(linspace(0,gra.T,gra.N),u(:,j),linspace(0,gra.T,gra.N),u(:,j+length(gra.spinlist)))
    xlabel('Time(Seconds)')
    ylabel('RF Power(Hz)')
    title(['GRAPE control Parameters Variation || Fidelity :',num2str(gra.IDEALfidelity),' || Channel : ',num2str(j)])
    grid on
end


