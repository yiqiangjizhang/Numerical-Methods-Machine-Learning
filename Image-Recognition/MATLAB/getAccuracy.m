function [accuracy] = getAccuracy(predictions,Y)
%{
Get the accuracy
%}

accuracy = sum(predictions == Y) / length(Y);

end
