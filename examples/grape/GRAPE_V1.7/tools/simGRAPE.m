function simGRAPE(GRinfo,indm,swh,T2)

if nargin<1
    disp('  ')
    disp('  ')
    disp('      simGRAPE(GRinfo,initialdm,swh,T2)')
    disp('      Simlulates the NMR spectra for the simulated gate')
    disp('      applied on some density matrix given by initialdm')
    disp('  ')
    disp('          initialdm - Initial Density Matrix')
    disp('                      {default: Equilibrium Density matrix taken}')
    disp('                swh - Spectral Width {default : 40000}')
    disp('                 T2 - T2 decay rate {default : 0.2}           ')
    disp('  ')
    disp('      SIMPLEST INPUT : simGRAPE(GRinfo)')
    disp('  ')
    disp('   It is unnecessary to provide all the inputs.  Empty inputs indicate')
    disp('   indicate default selections.')
    disp('  ')
    disp('   (Hemant Katiyar, 2012)')
    return
end
 
 
 global gra
 gra=GRinfo;
 [Ix,Iy,Iz,~,~,~,sIHz] = prodop(gra.spinNumbers,gra.spinlist);


%-------Declaring Default Inputs--------------------
if (nargin < 2 || isempty(indm)); indm=sum(Iz,3); 
end;
if (nargin < 3 || isempty(swh));  swh=40000; end;
if (nargin < 4 || isempty(T2));   T2=0.2; end;
 
%% CALCULATING TIME AND FREQUENCY AXIS
  
td = round(swh);
dw = 1/swh;
t  = 0:dw:(td-1)*dw;
k  = -td/2:(td-1)/2;
f  = k/(td*dw);


eham  = gra.Hint;
detop = sum(Ix,3)+1i*sum(Iy,3);


outdm_sim = gra.Usim*indm*(gra.Usim)';
outdm_th  = gra.Utarg*indm*(gra.Utarg)';

s_sim=zeros(1,td); s_th=zeros(1,td);
for m=1:td
 outdm1_sim=expm(-1i*eham*t(m))*outdm_sim*expm(1i*eham*t(m));
 outdm1_th=expm(-1i*eham*t(m))*outdm_th*expm(1i*eham*t(m));
 s_sim(m)=trace(detop*outdm1_sim);
 s_th(m)=trace(detop*outdm1_th);
end

s_sim=s_sim.*exp(-t/T2);
s_th=s_th.*exp(-t/T2);

%------------------Fourier Transform------------------------
S_sim=fftshift(fft(s_sim)); 
S_th=fftshift(fft(s_th));

figure
subplot(2,2,1),plot(f,real(S_th));
xlabel('Frequency in Hz'); ylabel('Intensity (S)');
title('Ideal (Real)');
subplot(2,2,2),plot(f,imag(S_th));
xlabel('Frequency in Hz'); ylabel('Intensity (S)');
title('Ideal (Imag)');

subplot(2,2,3),plot(f,real(S_sim));
xlabel('Frequency in Hz'); ylabel('Intensity (s)');
title(['Simulated (Real); Fidelity:',num2str(gra.IDEALfidelity)]);
subplot(2,2,4),plot(f,imag(S_sim));
xlabel('Frequency in Hz'); ylabel('Intensity (s)');
title(['Simulated (Imag): Fidelity:',num2str(gra.IDEALfidelity)]);

