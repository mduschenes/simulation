function  new_u = addGRAPE(GRinfo,noofadd)

if nargin<1
    disp(' ')
    disp(' new_u = addGRAPE(GRinfo,N)')
    disp('      add N GRAPE sequences')
    disp('  ')
    disp('             N - Number of times to add')
    disp('  ')
    disp('      SIMPLEST INPUT : ampwr(GRinfo,N)')
    disp('  ')
    disp('  ')
    disp('   (Hemant Katiyar, 2012)')
    return
end


global gra
gra=GRinfo;
del_t=GRinfo.del_t;
indel=GRinfo.initdelay;

% Find how many zero controls are needed
n=round(2*indel/del_t);

initial_u= GRinfo.u;


N=length(initial_u);
new_u(1:N,:)=initial_u;

for j=1:noofadd-1
    new_u(length(new_u)+1:length(new_u)+n,:)=0;
    new_u(length(new_u)+1:length(new_u)+N,:)=initial_u;
end
    