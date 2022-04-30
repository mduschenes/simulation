% Author : Hemant Katiyar (hkatiyar@uwaterloo)

% Modified from original code by Dr. T.S. Mahesh
% Only modification I did was change Iz and IHz to diagonal 
% and 2D matrix respectively

function [Ix,Iy,Iz,IHx,IHy,IHz,D] = prodop(spinNumbers,spinList)

if nargin < 2; spinList = ones(length(spinNumbers),1); end;

M = length(spinList);
N = sum(spinList);

spins = [];
for k = 1:M
  spins = [spins spinNumbers(k)*ones(1,spinList(k))];
end

D = prod(2*spins+1);

Ix = zeros(D,D,N); Iy = Ix;Iz = Ix;
% Iz = zeros(D,N); 

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
IHx = zeros(D,D,M); IHy = IHx; 
IHz = zeros(D,M); 
firstsp = 1;
for k = 1:M
  lastsp = firstsp + spinList(k) - 1;
  IHx(:,:,k) = sum(Ix(:,:,firstsp:lastsp),3);
  IHy(:,:,k) = sum(Iy(:,:,firstsp:lastsp),3);
  for j=firstsp:lastsp
	IHz(:,k) = IHz(:,k)+Iz(:,j);
  end
  firstsp = lastsp + 1;
end

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

