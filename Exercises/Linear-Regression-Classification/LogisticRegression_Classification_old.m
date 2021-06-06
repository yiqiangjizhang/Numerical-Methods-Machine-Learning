%-------------------------------------------------------------------------%
% Assignment 3: Linear Regression Classification using Machine Learning
%-------------------------------------------------------------------------%

% Date: 11/05/2021
% Author/s: 
%   Yi Qiang Ji
%   Biel Galiot
%   Cristian Asensio
%
% Subject: Machine Learning
% Professor: Alex Ferrer

close all;
clear all;

format long eng

% Set interpreter to latex
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');


% Load data
[Xdata,ydata] = loadData();

% plotData(Xdata, ydata);

[m, n] = size(Xdata);

X = computeFullX(Xdata);

theta0 = computeInitialTheta(n);

[cost, grad] = computeCostFunction(X, ydata,theta0);


[thetaOpt,allCost] = solveTheta(X,ydata,theta0);



plotData(X,ydata)

%figure
%plot(allCost)


function [theta,allCost] = solveTheta(X,y,theta0)
    
    theta = theta0';
    alpha = 10e-6;
    
    for i=1:1:1000
        [~,nablaJ] = computeCostFunction(X,y,theta');
        theta = theta - alpha.*nablaJ;
        if mod(i,10)==0
            figure(1)
        plotBoundary(X,y,theta)
        drawnow
        end
        
    end
    hold on
    allCost = zeros(100,1);
    
end


function plotBoundary(X,y,theta)
    x2 = (-theta(2)*X(:,1) - theta(1))/(theta(3));
    x1 = X(:,1);
    plot(x1,x2);
end

function [X,y] = loadData()
data = load('dataLogistic.txt');
X = data(:, [1, 2]); 
y = data(:, 3);
end

function plotData(X,y)
% Plot  with different colors

plot(X(:,1),X(:,2), '.');
for i=1:1:length(y)
    if y(i) == 1
        figure(1)
        plot(X(i,1),X(i,2), 'r+');
        hold on;
    else
        plot(X(i,1),X(i,2), 'bo');
        hold on;
    end
end

% figure(1)
% grid on;
% grid minor;
% xlabel('');
% ylabel();
% legend();
% title();
end


function X = computeFullX(X)
    X = [X ones(size(X,1),1)];
end

function theta0 = computeInitialTheta(n)
%     theta0 = ones(n+1,1)*10^(-3);
        theta0 =  2*ones(n+1,1)*10^(-3);
end

function [J,grad] = computeCostFunction(X,y,theta)
    m = size(y,1);
    
    h = hypothesisFunction(X,theta);
    g = sigmoid(h);
    J = m^(-1)*sum((1 - y(:)).*(-log10(1 - g')) ...
        + y(:).*(-log10(g')));
    
    grad = sum(m^(-1)*((g') - y).*X);
    
end

function h = hypothesisFunction(X,theta)
    h = X * theta;
end

function g = sigmoid(z)
    g = (1+exp(-z)).^(-1);
end