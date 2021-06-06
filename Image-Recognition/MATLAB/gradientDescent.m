function [W1, b1, W2, b2] = gradientDescent(X, Y, alpha, iterations) 
%{
Gradient Descent
%}
[W1, b1, W2, b2] = initParameters();

for i=1:1:iterations
    [Z1, A1, Z2, A2] = forwardPropagation(W1, b1, W2, b2, X);
    [dW1, db1, dW2, db2] = backwardPropagation(Z1, A1, A2, W1, W2, X, Y);
    [W1, b1, W2, b2] = updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha);
    if mod(1,10) == 0
            fprintf('Iteration: %f',i)
            predictions = getPredictions(A2);
            accuracy = getAccuracy(predictions,Y);
            disp(accuracy);
    end
end

end