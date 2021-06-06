% function LogisticRegression
clc
close all
clear all

[Xdata,ydata] = loadData();

plotData(Xdata, ydata);

[m, n] = size(Xdata); % m = 100, n = 2
X = computeFullX(Xdata);
theta0 = computeInitialTheta(n);
[cost, grad] = computeCostFunction(X, ydata,theta0,m);


[thetaOpt,allCost] = solveTheta(X,ydata,theta0,m);
plotBoundary(X,ydata,thetaOpt)
figure
plot(allCost)
%drawnow i plotboundary
% end

function [theta,allCost] = solveTheta(X,y,theta0,m)
    error = 1;
    de = 10^-7;
    alpha = 10^-4;
    theta = theta0;
    theta_prev = theta;
    i = 1;
    while abs(error) > de

        [allCost_var,grad_J] = computeCostFunction(X,y,theta,m);
        %      grad_J = 1/m.*sum(g-y(:).*X(:));
        theta = theta - alpha*grad_J;
        error = max(theta-theta_prev);
        allCost(i) = norm(allCost_var);
        theta_prev = theta;
%         if(rem(i,100)==0)
%             drawnow()
%             plot(allCost(1:i))
%         end
        i = i + 1;
        
    end
end


function plotBoundary(X,y,theta)
    x1 = X(:,1);
    x2 = X(:,2);
    cols = [255 0 0].*y;
    scatter(x1,x2,[],cols)
    hold
    syms x y2;
    y2 = (- theta(3) - theta(1) * X(1))/theta(2);
    fplot(y2);
end

function [X,y] = loadData()
data = load('dataLogistic.txt');
X = data(:, [1, 2]); 
y = data(:, 3);
end

function plotData(X,y)
% Plot  with different colors
    x1 = X(:,1);
    x2 = X(:,2);
    cols = [255 0 0].*y;
    scatter(x1,x2,[],cols)

end


function X = computeFullX(X)
    X(:,3) = 1;
end

function theta0 = computeInitialTheta(n)
    
    %theta0 = 0.1*rand(1,n+1);
    theta0 = zeros(1,n+1);
    %theta0 = -10.^6*rand(1,n+1);
    %theta_0 = 
end

function [J,grad_J] = computeCostFunction(X,y,theta,m)
    h = hypothesisFunction(X,theta);
    g = sigmoid(h);
    J = 1/m*sum((1-y(:)).*(-log10(1-g'))+y(:).*(-log10(g')));
    grad_J = zeros(1,3);
        for j=1:1:m
            grad_J = grad_J + (g(j)-y(j))*X(j,:);
        end
    grad_J = grad_J/m;
end

function h = hypothesisFunction(X,theta)
    h = theta*X';
end

function g = sigmoid(z)
    g = (1+exp(-z)).^(-1);
end