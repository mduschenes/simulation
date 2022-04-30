function multi_fac = quadratic_fit(y)

h=1;
A=[1/(2*h^2) -(1/h)^2 1/(2*h^2); -(1/h) (1/h) 0; 1 0 0];
A=A*y';
a=A(1); b=A(2);
% c=A(3);

multi=[0 1 2];

if (a>0 || a==0)
    [~, idx] = max(y); 
    [x_max] = ind2sub(size(y),idx);
    multi_fac = multi(x_max);
else
    multi_fac = -b/(2*a);
end

multi_fac=min(max(0.1,multi_fac),2);