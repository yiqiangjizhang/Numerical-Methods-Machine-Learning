function [W1, b1, W2, b2, W1_1, b1_1] = initParameters(Xtrain, nodes)
    %{
    Function to initialize parameters
    %}

    % For a single hidden layer
    W1 = rand(nodes, size(Xtrain, 2)) - 0.5;
    b1 = rand(nodes, 1) - 0.5;
    W2 = rand(10, nodes) -0.5;
    b2 = rand(10, 1) -0.5;
    
    W1_1 = rand(10, nodes) -0.5;
    b1_1 = rand(10, 1) -0.5;

end
