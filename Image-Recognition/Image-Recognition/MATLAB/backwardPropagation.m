function [dW1, db1, dW2, db2] = backwardPropagation(Z1, A1, A2, W2, Xtrain, Ytrain, j, W1, b1, b2, alpha)
    %{
    Backward propagation
    %}

    % Number of images
    m = length(Ytrain);

    one_hot_Y = onehot(Ytrain(j,:));

    delta_o = A2 - one_hot_Y;
    dZ2 = A2 - one_hot_Y(:, j);
    
    
    dW2 = 
    
    
    
    
    
    dW2 = 1 / m * dZ2 * A1';
    db2 = 1 / m * sum(dZ2);
    dZ1 = W2' * dZ2 * deriv_ReLU(Z1);
    % dZ1 = W2' * dZ2 * Sigmoid_deriv(Z1);
    dW1 = 1 / m * dZ1 * Xtrain(j, :);
    db1 = 1 / m * sum(dZ1);
    
    
    W1 = W1 - alpha * dW1;
    b1 = b1 - alpha * db1;
    W2 = W2 - alpha * dW2;
    b2 = b2 - alpha * db2;

end
