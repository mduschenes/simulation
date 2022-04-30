function [Ix,Iy,Iz,IHx,IHy,IHz,sIHz] = prodop(spinNumbers,spinList)

if nargin < 1;
disp(['  '])
disp(['   function [Ix,Iy,Iz,IHx,IHy,IHz,sIHz] = prodop(spinNumbers,spinList)'])
disp(['  '])
disp(['   Generates 3-dimensional arrays of x, y and z product operators '])
disp(['   for the spin system described by spinNumbers and spinList.'])
disp(['  '])
disp(['   For example, for a system of two spins 1/2 of different species '])
disp(['   spinNumbers = [1/2 1/2] and spinList = [1 1].  For a system of'])
disp(['   two spins 1/2 of same species, spinNumbers = [1/2] and spinList = [2].'])
disp(['   IHq(:,:,n) have sums of Iq for each nuclear species n.  sIHz '])
disp(['   is the sum of z-operators of all the spins.'])
disp(['  '])
disp(['   Defaults: If spinList is not given, a single spin for each entry in'])
disp(['   the spinNumbers is assumed.'])
disp(['  '])
disp(['   (T. S. Mahesh, 2006)'])
disp(['  '])
return
end

if nargin < 2; spinList = ones(length(spinNumbers),1); end;

M = length(spinList);
N = sum(spinList);

spins = [];
for k = 1:M
  spins = [spins spinNumbers(k)*ones(1,spinList(k))];
end

D = prod(2*spins+1);
Ix = zeros(D,D,N); Iy = Ix; Iz = Ix; 

id = eye(N,N);
for k=1:N
  Px=1; Py=1; Pz=1;
  for j=1:N
    [pe,px,py,pz] = genBasicOp(spins(j));
    Px = kron(Px,(id(k,j)*px + (1-id(k,j))*pe));
    Py = kron(Py,(id(k,j)*py + (1-id(k,j))*pe));
    Pz = kron(Pz,(id(k,j)*pz + (1-id(k,j))*pe));
  end
  Ix(:,:,k) = Px;
  Iy(:,:,k) = Py;
  Iz(:,:,k) = Pz;
end

% Heteronuclear sums
firstsp = 1;
for k = 1:M
  lastsp = firstsp + spinList(k) - 1;
  IHx(:,:,k) = sum(Ix(:,:,firstsp:lastsp),3);
  IHy(:,:,k) = sum(Iy(:,:,firstsp:lastsp),3);
  IHz(:,:,k) = sum(Iz(:,:,firstsp:lastsp),3);
  firstsp = lastsp + 1;
end

sIHz = sum(IHz,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Pe,Px,Py,Pz] = genBasicOp(j)

% function [Pe,Px,Py,Pz] = genBasicOp(spinNumber)
% Generates e (identity), x, y, z rotation operators of spin-j

Px = 0; Py = 0; Pz = 0;

if j < 0.5
  disp('spinNumber should be a positive integer or a positive half-integer')
  return
end

m = j:-1:-j;   % spin angular momentum eigenvalues
Pz = diag(m);  Pe = eye(size(Pz));
Pp = zeros(size(Pz));  Pm = Pp;  % Raising and Lowering ops.

% We have Pp|j,m> = Sqrt((j-m)(j+m+1)) |j,m+1>  and
%         Pm|j,m> = Sqrt((j+m)(j-m+1)) |j,m-1>
for k = 2:length(m); Pp(k-1,k) = sqrt((j-m(k))*(j+m(k)+1)); end;
for k = 1:length(m)-1; Pm(k+1,k) = sqrt((j+m(k))*(j-m(k)+1)); end;
Px = (Pp + Pm)/2;  Py = (Pp - Pm)/2i;

