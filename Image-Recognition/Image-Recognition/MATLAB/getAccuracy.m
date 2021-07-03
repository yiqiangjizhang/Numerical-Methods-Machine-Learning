function [accuracy] = getAccuracy(x_index, Ytrain)
    %{
    Get the accuracy
    %}

    % Loop through all the indices and if the index
    accuracy = sum((x_index - 1) == Ytrain) / length(Ytrain);

end
