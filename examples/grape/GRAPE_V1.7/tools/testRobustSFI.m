function testRobustSFI(GRinfo,inc_range,rang,sfi_range,state)

if nargin<1
    disp('  ')
    disp('  ')
    disp('      testRobust(GRinfo,inc_range,Range,SFImax_var)')
    disp('      Calculate the Robustness of the simulated gate under RF inhomogenity')
    disp('              ')
    disp('        inc_range - the increament value in spanning the range')
    disp('                     {default:0.01}')
    disp('            Range - range of RF inhomogenity e.g [.80 1.2]')
    disp('                    {default : takes the range for which pulse is generated}')
    disp('       SFImax_var - Max offset variation e.g 10 will make variation from -10 to 10')
    disp('                    {default : -10:10}')
    disp('  ')
    disp('      SIMPLEST INPUT : testRobust(GRinfo,[],[])')
    disp('  ')
    disp('  ')
    disp('   It is unnecessary to provide all the inputs.  Empty inputs indicate')
    disp('   indicate default selections.')
    disp('  ')
    disp('   (Hemant Katiyar, 2012)')
    return
end

global gra
gra=GRinfo;

%---------------Declaring Defaults------------
if (nargin < 2 | isempty(inc_range)); inc_range=.01; end
if (nargin < 3 | isempty(rang)); range=min(gra.rfINHrange):inc_range:max(gra.rfINHrange);
else
    range=min(rang):inc_range:max(rang);
end
if (nargin < 4 | isempty(sfi_range)); sfi=-10:10; sfi_range=10;
else
    sfi=-sfi_range:sfi_range;
end
if (nargin < 5 | isempty(state)); state='n'; end
%-----------------------------------------------------------


X1(:,:,1)=eye(2^gra.nspins);
U1=zeros(2^gra.nspins,2^gra.nspins,gra.N);
A=zeros(1,gra.N);
phi=zeros(1,gra.N);
spinlist = gra.spinlist;
u=gra.u;

J=gra.Couplings;

fidelity=zeros(length(range),length(sfi));

for s=1:length(sfi)
    fprintf('SFI iteration : %2g out of %2g \n',s,length(sfi))
    v1=gra.Chem_shift+sfi(s);
    Hint = generate_free_evolH(spinlist,v1,J);
    if state == 'y'
        gra.Ud = expm(1i*gra.Hint*gra.initdelay);
        gra.RHO_init = gra.Ud*gra.RHO_init*gra.Ud';
    else
        gra.U_target=expm(1i*gra.initdelay*gra.Hint)*gra.Utarg*expm(1i*gra.initdelay*gra.Hint);
        
    end
    for k=1:length(range)
        for j=1:gra.N
            sum_hamil=zeros(2^gra.nspins);
            for n=1:length(spinlist)
                A(j,n)=range(k)*sqrt(u(j,n)^2+u(j,n+length(spinlist))^2);
                phi(j,n)=atan2(u(j,n+length(spinlist)),u(j,n));
                sum_hamil = sum_hamil+A(j,n)*cos(phi(j,n))*(gra.Hrf{1,n}) + A(j,n)*sin(phi(j,n))*(gra.Hrf{1,n+length(spinlist)});
            end
            U1(:,:,j) = expm(-1i*(gra.T/gra.N)*(Hint + sum_hamil));
            X1(:,:,j+1)=U1(:,:,j)*X1(:,:,j);
        end
        if state=='y'
            fidelity(k,s) = (abs(trace(gra.RHOtarg'*X1(:,:,gra.N+1)*gra.RHO_init*X1(:,:,gra.N+1)')));
        else
            fidelity(k,s) = (abs(trace(gra.U_target'*X1(:,:,gra.N+1)))/2^(gra.nspins))^2;
        end
        
    end
end

avgfil=100*mean(fidelity(:,sfi_range+1));
fprintf('Average Fidelity : %2.6f\n',avgfil)
avgfil_SFI=100*mean(mean(fidelity));
fprintf('Average Fidelity SFI: %2.6f\n',avgfil_SFI)

%--------PLOT-----------------

figure
surf(sfi,range,fidelity)
xlabel('SF Inhomogenity'); ylabel('RF Inhomogenity'); zlabel('Fidelity')
title('Robustness Plot')
title(['Robustness Plot || Avg. Fidelity:',num2str(avgfil),'% || Avg. Fidelity SFI:',num2str(avgfil_SFI),'%']);
grid on
axis tight
