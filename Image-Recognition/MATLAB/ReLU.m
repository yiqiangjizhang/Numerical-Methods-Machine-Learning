function [Z] = ReLU(Z)
%{
Rectified linear activation function
%}

if Z < 0
    Z = 0;
end

end