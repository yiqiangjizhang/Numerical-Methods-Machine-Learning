function [Z1, A1, Z2, A2] = forwardPropagation(W1, b1, W2, b2, X)
%{
Forward propagation
%}

Z1 = W1 * X + b1;
A1 = ReLU(Z1);
Z2 = W2 * A1 + b2;
A2 = softmax(Z2);

end