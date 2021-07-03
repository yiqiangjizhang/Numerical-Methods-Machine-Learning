function [dh] = Sigmoid_deriv(A1)
    %{
    Derivative of Sigmoid
    %}

    dh = A1 .* (1 - A1);

end
