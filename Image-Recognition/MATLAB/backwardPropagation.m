function [dW1, db1, dW2, db2] = backwardPropagation(Z1, A1, A2, W2, X, Y )
%{
Forward propagation
%}

m = length(Y);

one_hot_Y = onehot(Y);

dZ2 = A2 - one_hot_Y;
dW2 = 1/m * dZ2 * A1';
db2 = 1/m * sum(dZ2);
dZ1 = W2' * dZ2 * deriv_ReLU(Z1);
dW1 = 1/m * dZ1*X';
db1 = 1/m * sum(dZ1);

end




