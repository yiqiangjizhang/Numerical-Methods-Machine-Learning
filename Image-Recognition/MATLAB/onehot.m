function [one_hot_Y] = onehot(Ytrain)
    %{
    OneHot encoder
    %}

    % one_hot_Y = zeros(length(Ytrain), max(Ytrain) + 1);
    one_hot_Y = zeros(10, 1);

    % for i = 1:1:size(Ytrain, 1)
    % one_hot_Y(i,Ytrain+1) = 1;

    if Ytrain == 0
        one_hot_Y(1) = 1;
    elseif Ytrain == 1
        one_hot_Y(2) = 1;
    elseif Ytrain == 2
        one_hot_Y(3) = 1;
    elseif Ytrain == 3
        one_hot_Y(4) = 1;
    elseif Ytrain == 4
        one_hot_Y(5) = 1;
    elseif Ytrain == 5
        one_hot_Y(6) = 1;
    elseif Ytrain == 6
        one_hot_Y(7) = 1;
    elseif Ytrain == 7
        one_hot_Y(8) = 1;
    elseif Ytrain == 8
        one_hot_Y(9) = 1;
    elseif Ytrain == 9
        one_hot_Y(10) = 1;
    end

    % end

    % one_hot_Y = one_hot_Y';
end
