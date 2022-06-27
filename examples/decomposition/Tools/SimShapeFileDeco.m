function DM = SimShapeFileDeco(filename,Length_pulse,dt,addphase,Ud,IHz,W1,W2,DM)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DECOHERENCE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_qubits = 4;
Dimension = 2^N_qubits;
% t2 = [1.09 .92 .84 .97];
% t2 = [1.27 1.17 1.19 1.13]*200;
% t1 = [4.6 4.6 4.8 7.7]*200;
t2 = [0.5673 1.0641 0.3181 0.4155 ];
t1 = [4.6 4.6 4.8 7.7 ];
decoherence_matrix = ones(Dimension);

for ii = 1:Dimension
    for jj = 1:Dimension
        row_state = dec2bin(ii-1,N_qubits);
        column_state = dec2bin(jj-1,N_qubits);
        for kk = 1:4
            if (row_state(kk)-column_state(kk))~=0
                decoherence_matrix (ii,jj) = decoherence_matrix(ii,jj)*exp(-dt/t2(kk));
            else
                decoherence_matrix (ii,jj) = decoherence_matrix(ii,jj)*exp(-dt/t1(kk));
            end
        end
    end
end

dec5us=decoherence_matrix;
% dec5us=ones(2^4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[power,phase]=dataout(filename,'Test',28,Length_pulse);

%convert to amp/phase
mxfq=(1/4/20e-6);
amp = power/100*(mxfq)*2*pi;
pha=pi/180*phase+addphase*pi/2;
np=Length_pulse;

k=1;
DM=Ud*DM*Ud';
for n=1:np
    AIz = amp(n)*IHz(:,k); PIz = pha(n)*IHz(:,k);
    U   = (exp(-1i*PIz).*exp(-1i*PIz)').*(W1*(exp(-1i*dt*AIz).*W2));
    DM  = dec5us.*U*DM*U';
end
DM=Ud*DM*Ud';
end