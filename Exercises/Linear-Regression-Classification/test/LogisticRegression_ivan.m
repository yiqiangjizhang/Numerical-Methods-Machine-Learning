%-------------------------------------------------------------------------%
% Linear Regression
%-------------------------------------------------------------------------%

% Date: 11/05/2021

% Author/s: Ivan Sermanoukian Molina
%           Álvaro Sánchez del Rio
%           Lluc-Ramon Busquets Soler
%           Javier Roset Cardona

% Subject: Numerical Tools in Machine Learning for Aeronautical Engineering
% Professor: Alex Ferrer Ferre

% Clear workspace, command window and close windows
clc
clear all
close all

% LaTeX configuration
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

%% Apartat 1. Cas lineal

% Load data
[Xdata,ydata] = loadData();
percentage = 0.2;

plotData(Xdata, ydata);

[m, n] = size(Xdata); % m = 100, n = 2
X = computeFullX(Xdata);

percentage = 0.2;
[Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(X,ydata,m,percentage);

[m_t, n_t] = size(Xtrain);

theta0 = computeInitialTheta(n);
[cost, grad] = computeCostFunction(Xtrain, Ytrain,theta0,m_t);

[thetaOpt,allCost] = solveTheta(Xtrain,Ytrain,theta0,m_t);
plotBoundary(Xtrain,Ytrain,thetaOpt)
% figure
plot(allCost)
xlim([30 100]);
ylim([30 100]);
%drawnow i plotboundary
% end

function [theta,allCost] = solveTheta(X,y,theta0,m)
    error = 1;
    de = 10^-8;
    alpha = 10^-3;
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
        if(rem(i,10000)==0)
            drawnow()
            plot(allCost(1:i))
        plotBoundary(X,y,theta);
        end
        
        i = i + 1;
        
    end
    % plotBoundary(X,y,theta);
end


function plotBoundary(X,y,theta)
    x1 = X(:,1);
    x2 = X(:,2);
    cols = [255 0 0].*y;
    figure(2)
    clf;
    scatter(x1,x2,[],cols)
    hold on;
    syms x y2;
    y2 = (- theta(3) - theta(1) * x)/theta(2);
    fplot(y2,'b');
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

function [Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(xdata,ydata,nData,percentage)

    nTest = round(percentage*nData);
    index = randperm(nData,nTest);
    
    % nTrain = nData - nTest;
    
    Xtest = xdata(index,:);
    Ytest = ydata(index);
    
    xdata(index,:) = [];
    ydata(index) = [];
    Xtrain = xdata;
    Ytrain = ydata;

end