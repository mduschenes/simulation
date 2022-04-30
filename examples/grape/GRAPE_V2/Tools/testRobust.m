function testRobust(InputFile)

SavePath = [pwd filesep 'SaveOutputs' filesep 'SaveOutputsGRAPE' filesep];
% SavePath = [pwd filesep 'SaveOutputs' filesep 'SaveOutputsSMOOTH' filesep];

load([SavePath InputFile],'-mat','GR','u','Mol');

inc_range=.01;

% range=min(GR.rfi(:,1)):inc_range:max(GR.rfi(:,1));
range = 0.95:inc_range:1.05;


[Ix,Iy,Iz,~,~,IHz,D] = prodopSparse(Mol.spinNumbers,Mol.spinlist);
Hint = genHint(Mol.spinlist,Mol.v,Mol.J,D,Ix,Iy,Iz);

% Some operators for avoiding expm
Had = (1/sqrt(2))*[1 1; 1 -1];
for j=2:Mol.nspins
    Had=kron(Had,(1/sqrt(2))*[1 1; 1 -1]);
end

UH0 = expm(-1i*Hint*GR.del_t/2);
W1 = UH0*Had;
W2 = Had*UH0;

% Negative Evolution Unitary
Ud = expm(1i*GR.initdelay*Hint);

% Unitary for which GRAPE will be optimized
UTopt = Ud*GR.Utarg*Ud;

Fid=zeros(length(range),1);
for k=1:length(range)
    Fid(k) = CalcFidelity(u,Mol.nspins,GR.N,GR.m,[range(k) 1],IHz,W1,W2,UTopt,GR.del_t);
end

avgfil=100*mean(Fid);
fprintf('Average Fidelity : %2.6f\n',avgfil)

%--------PLOT-----------------
figure
plot(range,Fid)
xlabel('RF Inhomogenity'); ylabel('Fidelity');
title('Robustness Plot')
title(['Robustness Plot || Avg. Fidelity:',num2str(avgfil),'%']);
grid on
axis tight
