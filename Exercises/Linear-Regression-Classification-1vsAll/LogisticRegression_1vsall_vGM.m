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

load iris.mat

% Load data
[Xdata,ydata] = loadData();
% ydata(:,2) = randperm(ydata(:,1));
% ydata(:,3) = randperm(ydata(:,1));

%figure(1)
%plotData(Xdata, ydata);

Xdata_plot = meas(:,1:4);
Ydata_plot = meas(:,end);

xnames = {'sepal length'; 'sepal width'; 'petal length';'petal width'};
ynames = {'sepal length'; 'sepal width'; 'petal length';'petal width'};
gplotmatrix(Xdata_plot,Xdata_plot,Ydata_plot,'brg','.ox',[],'on',[],xnames,ynames);

[m, n] = size(Xdata);

d=2;
X = computeFullX(Xdata,d);

nData = size(Xdata,1);

[Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(X,ydata,nData);

[m_train, n_train] = size(Xtrain);

theta0 = computeInitialTheta(size(X,2),size(Ytrain,2));

lambda = 1e-4;

[cost, grad] = computeCostFunction(Xtrain,Ytrain,theta0,d,lambda);

[thetaOpt,allCost] = solveTheta(Xtrain,Ytrain,theta0,d,lambda);


function [theta,fval,exitflag,output] = solveTheta(X,y,theta0,d,lambda)
theta0 = theta0(:);
options = optimoptions(@fminunc,'PlotFcn','optimplotfval','Algorithm','quasi-newton');
F =@(theta) computeCostFunction(X,y,theta,d,lambda); %num +  vect
[theta,fval,exitflag,output] = fminunc(F,theta0,options); %theta0 ha de ser vector
%figure(2)
plotBoundary(X,y,theta,d);
% drawnow
end
    
function plotBoundary(X,y,theta,d)
    close 
    x1 = X(:,1);
    x2 = X(:,2);
    % plot(x1,x2);
    cols = [255 120 200].*y;
    clear axes
    %figure(2)
    %scatter(x1,x2,[],cols)
    %xlabel('Speal width')
    %ylabel('Sepal length')
    hold on;
    
    X1v = linspace(min(x1), max(x1), 150)';
    X2v = linspace(min(x2), max(x2), 150)';
    
    %A = computeFullX([X1v X2v],d);
    %computeFullX(X2v,d);
    
    for i=1:length(X1v)
        XX = X1v(i);
        for j=1:length(X2v)
          YY = X2v(j);
          for k=1:size(y,2)
          A = (computeFullX([XX YY],d));
          %A = repmat(A,1,3);
          C = hypothesisFunction(A,theta,y);
          B(j,i,k) = C(k);
          end
        end
    end
    
    for k=1:size(y,2)
    figure(k+3)
    contour(X1v,X2v,B(:,:,k),[0 0]);
    hold on 
    scatter(x1,x2,[],cols)
    xlabel('Speal width')
    ylabel('Sepal length')
    title("Classification of type " + k);
    end
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

function theta0 = computeInitialTheta(n,ncoly)
    theta0 = zeros(n, ncoly)*10^(-7);
    theta0 = theta0(:); 
end

function [J,grad] = computeCostFunction(X,y,theta,d,lambda) %entrar com a vector
    
    m = size(y,1);
    h = hypothesisFunction(X,theta,y);
    g = sigmoid(h);


    %J = m^(-1)*sum( sum( (1 - y(:)).*(-log(1 - g)) ...
    %    + y(:).*(-log(g)) )) + 0.5*lambda*theta'*theta;
    
     J = 100*m^(-1)*sum( sum( (1 - y).*(-log(1 - g)) ...
        + y.*(-log(g)) )) + lambda*norm(theta)^2;%0.5*lambda*theta'*theta;
    
    %%%%%CORREGIR%%%%
    %theta = reshape(theta,[size(X,2) size(y,2)]);
    grad = m^(-1)*X'*((g) - y);% + (lambda*theta);
    grad = grad(:);
    
    %grad = grad(:,1);%???

end

function h = hypothesisFunction(X,theta,y)
    theta = reshape(theta,[size(X,2) size(y,2)]);
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