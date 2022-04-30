% function simbruk(GRinfo,shapename,sname,ext)
clc
clear all
load ala_90x112.mat
load ala_90x112_11.abc
dat = ala_90x112_11;

[Ix,Iy,Iz,~,~,~,sIHz] = prodop(GRinfo.spinNumbers,GRinfo.spinlist);

T2 = 0.2; 
swh=40000;
td=40*1024;
 
%CALCULATING TIME AND FREQUENCY AXIS
   
dw=1/swh;
t=0:dw:(td-1)*dw;
k=-td/2:(td-1)/2;
f=k/(td*dw);

detop=sum(Ix,3) + 1i*sum(Iy,3);

mxfq=(1/4/15.3e-6);                          % Max freq(amp) of the soft pulse
tp=GRinfo.T;                                    % Time duration of the soft pulse
amp = dat(:,1)/100*(mxfq)*2*pi;
pha=pi/180*dat(:,2);
% [a b]=find(pha>pi);
% pha(a,b) = pha(a,b)-2*pi;
% pha=mod(pi-pha,2*pi);

np=length(amp);
dt=tp/np;

indm = (Iz(:,:,1)+Iz(:,:,2)+Iz(:,:,3));

U=eye(8);
H=eye(8);

Hint = GRinfo.Hint;
for n=1:np
    H=Hint+amp(n)*(cos(pha(n))*GRinfo.Hrf{1}+sin(pha(n))*GRinfo.Hrf{2});
    U=expm(-1i*H*dt)*U;
end

Ud = expm(-1i*Hint*5e-6);
U=Ud*U*Ud;
outdm1=U*indm*U';
Fidelity = trace(Utarg'*U)/8;


 for m=1:td
     outdm18=expm(-1i*Hint*t(m))*outdm1*expm(1i*Hint*t(m));
     s(m)=trace(detop*outdm18);
 end
 
s=s.*exp(-t/T2);
 
S=fftshift(fft(s)); % Fourier Transform
figure
subplot(2,1,1),plot(t,real(s));
xlabel('Time in second'); ylabel('Intensity (s)');
title('Simulation of a three-spin system');
subplot(2,1,2),plot(f,imag(S));
xlabel('Frequency in Hz'); ylabel('Intensity (S)');
