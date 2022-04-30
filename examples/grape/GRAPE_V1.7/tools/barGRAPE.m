function barGRAPE(GRinfo,indm)

if nargin<1
    disp('  ')
    disp('      barGRAPE(GRinfo,initialdm)')
    disp('      Plots 3D barplot for the simulated gate')
    disp('      applied on some density matrix given by initialdm')
    disp('  ')
    disp('          initialdm - Initial Density Matrix')
    disp('                      {default: Equilibrium Density matrix taken}')
    disp('  ')
    disp('      SIMPLEST INPUT : barGRAPE(GRinfo)')
    disp('  ')
    disp('   It is unnecessary to provide all the inputs.  Empty inputs indicate')
    disp('   indicate default selections.')
    disp('  ')
    disp('   (Hemant Katiyar, 2012)')
    return
end
 
 
 global gra
 gra=GRinfo; 
 [~,~,Iz,~,~,~,sIHz] = prodop(gra.spinNumbers,gra.spinlist);

 %-------Declaring Default Inputs--------------------
if (nargin < 2 || isempty(indm)); indm=sum(Iz,3); end;

%-----------------------------------------------------
outdm_sim = gra.Usim*indm*(gra.Usim)';
outdm_th  = gra.Utarg*indm*(gra.Utarg)';

%--------PLOTTING------------------------------
figure
subplot(2,2,1),bar3(real(outdm_th));
title('Ideal (Real)');
subplot(2,2,2),bar3(imag(outdm_th));
title('Ideal (Imag)');

subplot(2,2,3),bar3(real(outdm_sim));
title(['Simulated (Real); Fidelity:',num2str(gra.IDEALfidelity)]);
subplot(2,2,4),bar3(imag(outdm_sim));
title(['Simulated (Imag): Fidelity:',num2str(gra.IDEALfidelity)]);


