function [deriv_Z] = deriv_ReLU(Z)
%{
ReLU derivative
%}
if Z<0
    deriv_Z = 0;
else
    deriv_Z = 1;
end
end