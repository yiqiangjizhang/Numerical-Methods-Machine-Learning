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
% ydata(:,2) = randperm(ydata(:,1));
% ydata(:,3) = randperm(ydata(:,1));

%figure(1)
%plotData(Xdata, ydata);

[m, n] = size(Xdata);

% Order 

d=2;

X = computeFullX(Xdata,d);

nData = size(Xdata,1);
[Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(X,ydata,nData);

[m_train, n_train] = size(Xtrain);

theta0 = computeInitialTheta(size(X,2),d);

lambda = 1e-4;
[cost, grad] = computeCostFunction(Xtrain,Ytrain,theta0,d,lambda);

[thetaOpt,allCost] = solveTheta(Xtrain,Ytrain,theta0,d,lambda);


function [theta,fval,exitflag,output] = solveTheta(X,y,theta0,d,lambda)

options = optimoptions(@fminunc,'PlotFcn','optimplotfval','Algorithm','quasi-newton');
F =@(theta) computeCostFunction(X,y,theta,d,lambda);
[theta,fval,exitflag,output] = fminunc(F,theta0,options);
%figure(2)
plotBoundary(X,y,theta,d);
drawnow
end
    
function plotBoundary(X,y,theta,d)
    close 
    x1 = X(:,1);
    x2 = X(:,2);
    plot(x1,x2);
    cols = [255 120 200].*y;
    clear axes
    figure(2)
    scatter(x1,x2,[],cols)
    hold on;
    
    X1v = linspace(min(x1), max(x1), 150)';
    X2v = linspace(min(x2), max(x2), 150)';
    
    %A = computeFullX([X1v X2v],d);
    %computeFullX(X2v,d);
    
    for i=1:length(X1v)
        XX = X1v(i);
        for j=1:LENGTH(x2V)
          YY = X2v(j);
          A = computeFullX([XX YY],d);
          B(j,i)=hypothesisFunction(A,theta);
        end
    end
    figure(3)
    contour(X1v,X2v,B,[0 0]);
    hold off
    %y2 = (-theta(1)*X(:,1) - theta(3))/(theta(2));
    %plot(X(:,1),y2);
end

function [X,y] = loadData()
%data = load('dataLogistic.txt');
%data = load('microchip.txt');
%data = load(iris.m);
load iris.mat
data = meas;

X = data(:, [1, 2]);%, 3, 4]);

y = zeros(size(data,1),3);

for i=1:size(data,1)
    if data(i,end)==1
        y(i,1)=1;
    elseif data(i,end)==2 
        y(i,2)=1;
    else
        y(i,3)=1;
    end


end
end

function plotData(X,y)
% Plot  with different colors
    
    % First and second
    x1 = X(:,1);
    x2 = X(:,2);
    
    % Column
    cols = [255 120 200].*y;
    
    scatter(x1,x2,[],cols)
    
end


function X = computeFullX(X,d)

if d==2
X1 = X(:,1);
X2 = X(:,2);
X1X2 = X(:,1).*X(:,2);
X1_2 = X(:,1).^2;
X2_2 = X(:,2).^2;

    X = [X1 X2 X1X2 X1_2 X2_2 ones(size(X,1),1)];
else
    X = [X ones(size(X,1),1)];
end
end

function theta0 = computeInitialTheta(n,d)
    theta0 = zeros(n, d);
%     theta0 = [-10; 0.1; 0.1];
end

function [J,grad] = computeCostFunction(X,y,theta,d,lambda)
    m = size(y,1);
    h = hypothesisFunction(X,theta);
    g = sigmoid(h);
    
    fullTheta=reshape(theta,[],1); % convert matrix to column vector
    
    %J = m^(-1)*sum( sum( (1 - y(:)).*(-log(1 - g)) ...
    %    + y(:).*(-log(g)) )) + 0.5*lambda*theta'*theta;
    
     J = m^(-1)*sum( sum( (1 - y).*(-log(1 - g)) ...
        + y.*(-log(g)) )) + lambda*norm(fullTheta)^2;%0.5*lambda*theta'*theta;
    
    %%%%%CORREGIR%%%%
    grad = m^(-1)*X'*((g) - y) + (lambda*fullTheta)';
    
    %grad = grad(:,2);???

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
    %Ytest = zeros()
    Ytest = ydata(index,:);
    
    xdata(index,:) = [];
    ydata(index,:) = [];
    Xtrain = xdata;
    Ytrain = ydata;

end