function dec5us = DecoherenceMatrix(N_qubits,t1,t2,dt)
Dimension = 2^N_qubits;
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
end