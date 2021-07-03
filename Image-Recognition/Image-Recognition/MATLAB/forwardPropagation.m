function [Z1, A1, Z2, A2] = forwardPropagation(W1, b1, W2, b2, Xtrain)
    %{
    Forward propagation
    %}

    Z1 = W1 * Xtrain + b1;
    A1 = ReLU(Z1);
    % A1 = Sigmoid(Z1);
    Z2 = W2 * A1 + b2;
    % A2 = softmax(Z2);
    A2 = Sigmoid(Z2);

end
