function Version_History
if nargin<1
    disp(' ')
    disp('------------------------------- Version 1.7 -------------------------------')
    disp(' ')
    disp(' Unnecesary calculation found and removed in calculation of new epsilon')
    disp(' Particularly epsi = 0 calculation removed & if multi_fac is equal to 1 or 2 then ')
    disp(' recalculation of robust fidelity removed')
    disp(' ')
    disp('------------------------------- Version 1.6 -------------------------------')
    disp(' ')
    disp(' Parallel Processing Addded - Gradient and expm calculation made parallel')
    disp(' Command to run parallel code --> GRinfo=rungrape_parallel(''InputFileName'')')
    disp(' Amplitude and Phase unnecessary storage removed - Thanks Ravi for pointing it out.')
    disp(' ')
    disp('------------------------------- Version 1.5 to Version 1.1 -------------------------------')
    disp(' ')
    disp(' Sorry, Forgot to keep the log')
end