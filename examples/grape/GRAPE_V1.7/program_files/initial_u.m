function u = initial_u(u_max,R)
global gra

u = zeros(gra.N,gra.m);
for j=1:gra.m
    u(:,j) = rand_u(u_max,R);
end



%****************************************************

function u_r= rand_u(S,R)
global gra

x=[1 R:R:gra.N-1 gra.N];
y=[0 S*(rand(1,length(x)-2)-.5) 0];
xx = 1:1:gra.N;
u_r = spline(x,y,xx);