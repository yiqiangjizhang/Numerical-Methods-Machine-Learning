function [W1, b1, W2, b2] = initParameters()
%{
Function to initialize parameters
%}

W1 = rand(1000,1);
b1 = rand(10,1);
W2 = rand(10,10);
b2 = rand(10,1);

end