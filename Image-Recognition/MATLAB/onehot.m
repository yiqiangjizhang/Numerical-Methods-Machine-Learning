function [one_hot_Y] = onehot(Y)
%{
OneHot encoder
%}

one_hot_Y = zeros(length(Y),max(Y)+1);
one_hot_Y(sort(Y), Y) = 1;
one_hot_Y = one_hot_Y';
end