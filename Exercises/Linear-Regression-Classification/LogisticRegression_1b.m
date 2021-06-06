%-------------------------------------------------------------------------%
% Assignment 3: Linear Regression Classification using Machine Learning
%-------------------------------------------------------------------------%

% Date: 11/05/2021
% Author/s: 
%   Yi Qiang Ji
%   Biel Galiot
%   Cristian Asensio
%
% Subject: Numerical Tools in Machine Learning for Aeronautical Engineering
% Professor: Alex Ferrer

clc;
close all;
clear all;

% Set interpreter to latex
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');


% Load data
[Xdata,ydata] = loadData();
figure(1)
plotData(Xdata, ydata);

[m, n] = size(Xdata);

X = computeFullX(Xdata);


nData = size(Xdata,1);
[Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(X,ydata,nData);

[m_train, n_train] = size(Xtrain);

theta0 = computeInitialTheta(n);

[cost, grad] = computeCostFunction(Xtrain,Ytrain,theta0);

[thetaOpt,allCost] = solveTheta(Xtrain,Ytrain,theta0);


function [theta,fval,exitflag,output] = solveTheta(X,y,theta0)
options = optimoptions(@fminunc,'PlotFcn','optimplotfval','Algorithm','quasi-newton');
F =@(theta) computeCostFunction(X,y,theta);
[theta,fval,exitflag,output] = fminunc(F,theta0,options);
figure(2)
plotBoundary(X,y,theta);
drawnow
end
    
function plotBoundary(X,y,theta)
    x1 = X(:,1);
    x2 = X(:,2);
    plot(x1,x2);
    cols = [255 0 0].*y;
    scatter(x1,x2,[],cols)
    hold on;
    y2 = (-theta(1)*X(:,1) - theta(3))/(theta(2));
    plot(X(:,1),y2);
end

function [X,y] = loadData()
data = load('dataLogistic.txt');
X = data(:, [1, 2]); 
y = data(:, 3);
end

function plotData(X,y)
% Plot  with different colors
    
    % First and second
    x1 = X(:,1);
    x2 = X(:,2);
    
    % Column
    cols = [255 0 0].*y;
    
    scatter(x1,x2,[],cols)
    
end


function X = computeFullX(X)
    X = [X ones(size(X,1),1)];
end

function theta0 = computeInitialTheta(n)
    theta0 = zeros(n+1, 1)*10^(-3);
%     theta0 = [-10; 0.1; 0.1];
end

function [J,grad] = computeCostFunction(X,y,theta)
    m = size(y,1);
    h = hypothesisFunction(X,theta);
    g = sigmoid(h);
    
    J = m^(-1)*sum((1 - y(:)).*(-log(1 - g)) ...
        + y(:).*(-log(g)));
    
    grad = m^(-1)*X'*((g) - y);
    
end

function h = hypothesisFunction(X,theta)
    h = X*theta;
end

function g = sigmoid(z)
    g = (1+exp(-z)).^(-1);
end

function [Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(xdata,ydata,nData)
    
    % Split data
    % Holdout validation percentage
    h_validation = 0.25; % Percentage
    
    test_data = round(h_validation*nData);
    
    index = randperm(nData,test_data);
    Xtest = xdata(index,:);
    Ytest = ydata(index);
    
    xdata(index,:) = [];
    ydata(index) = [];
    Xtrain = xdata;
    Ytrain = ydata;

end