clc 
clear

% [Ix,Iy,Iz,IHx,IHy,IHz,D] = prodopSparse([0.5],[5]);
% syms a b c j12 j13 Ja Jb Jc j23
% A = a*Iz(:,1)+a*Iz(:,2)+a*Iz(:,3)+b*Iz(:,4)+c*Iz(:,5)+...
%     j12*Iz(:,1).*Iz(:,2)+j13*Iz(:,1).*Iz(:,3)+Ja*Iz(:,1).*Iz(:,4)+Jb*Iz(:,1).*Iz(:,5)+...
%     j23*Iz(:,2).*Iz(:,3)+Ja*Iz(:,2).*Iz(:,4)+Jb*Iz(:,2).*Iz(:,5)+...
%     Ja*Iz(:,3).*Iz(:,4)+Jb*Iz(:,3).*Iz(:,5)+...
%     Jc*Iz(:,4).*Iz(:,5);
% 
% for k=1:2^3
%     A(4*(k-1)+1)-A(4*(k-1)+3)
%     A(4*(k-1)+2)-A(4*(k-1)+4)
% end
%     
%  dec2bin(0:2^5-1)


[Ix,Iy,Iz,IHx,IHy,IHz,D] = prodopSparse([0.5],[4]);
syms a b j12 j13 J j23
A = a*Iz(:,1)+a*Iz(:,2)+a*Iz(:,3)+b*Iz(:,4)+...
    j12*Iz(:,1).*Iz(:,2)+j13*Iz(:,1).*Iz(:,3)+J*Iz(:,1).*Iz(:,4)+...
    j23*Iz(:,2).*Iz(:,3)+J*Iz(:,2).*Iz(:,4)+...
    J*Iz(:,3).*Iz(:,4);

% for k=1:2^3
%     A(4*(k-1)+1)-A(4*(k-1)+3)
%     A(4*(k-1)+2)-A(4*(k-1)+4)
% end
    
for k=1:2^3
    A(2*k-1)-A(2*k)
end
 dec2bin(0:2^4-1)