function [h] = Sigmoid(Z)
    %{
    Sigmoid function
    %}

    h = 1 ./ (1 + exp(-Z));

end
