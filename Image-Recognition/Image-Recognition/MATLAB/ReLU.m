function [Z] = ReLU(Z)
    %{
    Rectified linear activation function
    %}

    Z = max(Z, 0);

end
