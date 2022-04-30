function U = SimUnitary(InputFile,IHx,IHy,Hint,PhaseShift,Ud)

% pathtosave= '/Users/hemantkatiyar/Google Drive/Documents/MATLAB/Compiler_strong_Editing/SaveOutputs/SaveOutputsGRAPE/';
pathtosave= 'G:\My Drive\Documents\MATLAB\Compiler_strong_Editing/SaveOutputs/SaveOutputsGRAPE/';

load([pathtosave InputFile],'-mat','GR','u','Mol');

U = eye(2^Mol.nspins);
for j=1:GR.N
    Hrf=zeros(2^Mol.nspins);
    for k=1:GR.m/2
        A = sqrt(u(j,2*k-1)^2+u(j,2*k)^2);
        P = atan2(u(j,2*k),u(j,2*k-1))+PhaseShift*pi/2;
        Hrf = Hrf + A*(cos(P)*IHx(:,:,k)+sin(P)*IHy(:,:,k));
    end
    U = expm(-1i*GR.del_t*(Hint+Hrf))*U;
end

U=Ud*U*Ud;