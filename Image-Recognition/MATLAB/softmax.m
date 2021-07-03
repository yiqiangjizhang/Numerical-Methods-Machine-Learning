function [Z] = softmax(Z)
    %{
    Softmax activation function
    %}

    Z = exp(Z) ./ sum(exp(Z));

end
