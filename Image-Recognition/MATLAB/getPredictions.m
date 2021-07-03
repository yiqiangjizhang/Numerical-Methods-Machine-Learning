function [x_index] = getPredictions(A2)
    %{
    Get the prediction
    %}

    [~, x_index] = max(A2, [], 2);

end
