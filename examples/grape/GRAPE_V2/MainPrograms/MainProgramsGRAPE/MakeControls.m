function u = MakeControls(R,N,m,maxAmp,PenaltyRange)

u=zeros(N,m);

for j=1:m
    U_penal=PenaltyRange(:,ceil(j/2));
    L_penal=-U_penal;
%     u(:,j) = transpose(rand_u(R,N,maxAmp(ceil(j/2))/2));
    u(:,j) = max(min(U_penal,transpose(rand_u(R,N,maxAmp(ceil(j/2))/2))),L_penal);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function u1_r= rand_u(R,N,maxAmp)

x=[1 R:R:N-1 N];
y=[0 (0.95*maxAmp/sqrt(2))*(rand(1,length(x)-2)-0.5) 0];
xx = 1:1:N;
u1_r = (spline(x,y,xx));

return
