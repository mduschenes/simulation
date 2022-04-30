function u_smooth = smoothGRAPE(GRinfo,del_t,n)

if nargin < 1
    disp('  ')
    disp('      u_new = smoothGRAPE(GRinfo,del_t,n)')
    disp('      Calculates  a smooth pulse sequence by interpolating the graph usinh spline')
    disp('  ')
    disp('               del_t(in us) - supply as 10 for 10e-6')
    disp('                          n - number of points you want to divide del_t in. ')
    disp('  ')
    disp('      SIMPLEST INPUT : u_new = smoothGRAPE(GRinfo,del_t,n)')
    disp('')
    disp('  ')
    disp('   (Hemant Katiyar, 2012)')
    return
end

global gra
gra = GRinfo;


for j=1:gra.m
    x=1:del_t:del_t*gra.N;
    y=gra.u(:,j);
    xx =1:del_t/n:del_t*gra.N;
    u_smooth(:,j) = spline(x,y,xx);
end
