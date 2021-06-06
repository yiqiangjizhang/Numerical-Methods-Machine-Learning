function [all_theta] = oneVsAll(X, y, num_labels, lambda)
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
    
% Set options for fminunc
    options = optimset('GradObj', 'on', 'MaxIter', 50);

% Run fmincg to obtain the optimal theta
% This function will return theta and the cost 
for i=1:num_labels
    %Set Initial theta
    initial_theta = zeros(n + 1, 1);
    [theta] = ...
        fmincg (@(t)(lrCostFunction(t, X, (y == i), lambda)), ...
                initial_theta, options);

    all_theta(i,:)=theta';
end

end
