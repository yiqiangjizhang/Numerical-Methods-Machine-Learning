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

% figure(2)
% plotBoundary(X,Ytrain,thetaOpt);

figure(3)
plot(allCost);


function [theta,allCost] = solveTheta(X,y,theta0)
    
    theta = theta0;
    theta_prev = theta;
    alpha = 1e-3;
    error = 1;
    threshold = 1e-8;
    i = 1;
    
%     % First Method: Set a threshold
%     while abs(error) > threshold
%         [allCost_temp,nablaJ] = computeCostFunction(X,y,theta);
%         theta = theta - alpha.*nablaJ;
%         error = max(theta-theta_prev);
%         allCost(i) = allCost_temp;
%         theta_prev = theta;
%         
%         Real time plot
%         if (mod(i,1000) == 0)
%             figure(2)
%             clf
%             plotBoundary(X,y,theta);
%             drawnow
%         end
%         i = i + 1;
%     end
    
    % Second Method: Set n-iterations
%     n_iter = 1e6;
%     for i=1:n_iter
%         [allCost_temp,nablaJ] = computeCostFunction(X,y,theta);
%         theta = theta - alpha.*nablaJ;
%         error = max(theta-theta_prev);
%         allCost(i) = allCost_temp;
%         theta_prev = theta;
%         
% %         Real time plot
% %         if (mod(i,1000) == 0)
% %             figure(2)
% %             clf
% %             plotBoundary(X,y,theta);
% %             drawnow
% %         end  
%     end
%     
%     figure(2)
%     plotBoundary(X,y,theta);
%     drawnow
%     
    % Cost function vs iteration using fminunc
    n_iter = 1e6;
    for i=1:n_iter
        
        [allCost_temp,nablaJ] = computeCostFunction(X,y,theta);
        % theta = theta - alpha.*nablaJ;
        fun = @(theta) theta - alpha.*nablaJ;
        
        x = fminunc(fun,theta0);
        error = max(theta-theta_prev);
        allCost(i) = allCost_temp;
        theta_prev = theta;
        
%         Real time plot
%         if (mod(i,1000) == 0)
%             figure(2)
%             clf
%             plotBoundary(X,y,theta);
%             drawnow
%         end  
    end
    
    
    
    
    
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

function [J,grad] = computeCostFunction_fminunc(X,y,theta0)
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