% Author : Hemant Katiyar, 30-June-2020, (hkatiyar@uwaterloo.ca)
%
% Helper function to generate a random guess
% free evolution values are in units of radians, hence divided by pi

function x0 = InitialGuess(TotVar,nSec,VarPerSec,DelayControl,MaxIniDelay)

x0 = rand(1,TotVar)*pi*1;

for ttt = 1:nSec
    x0(VarPerSec*ttt) = x0(VarPerSec*ttt)*5[[[[[[[[[[[[[[/1;
end

TimeOfTotalEvol=0;
for j = 1:nSec
    TimeOfTotalEvol = abs(x0(VarPerSec*j))+ TimeOfTotalEvol;
end
TimeOfTotalEvol = TimeOfTotalEvol*DelayControl/pi;
% MaxIniDelay
while TimeOfTotalEvol*1e+3>MaxIniDelay
    x0 = rand(1,TotVar)*pi*1;
    for ttt = 1:nSec
    x0(VarPerSec*ttt) = x0(VarPerSec*ttt)*8/1;
    end
    TimeOfTotalEvol=0;
    for j = 1:nSec
        TimeOfTotalEvol = abs(x0(VarPerSec*j))+ TimeOfTotalEvol;
    end
    TimeOfTotalEvol = TimeOfTotalEvol*DelayControl/pi;
end

end
