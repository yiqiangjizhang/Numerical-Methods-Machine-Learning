%% Image recognition

%-------------------------------------------------------------------------%
% Using MNIST Set
%-------------------------------------------------------------------------%

% Date: 08/06/2021
% Author/s: Yi Qiang Ji Zhang
% Subject: Numerical methods for Machine Learning
% Professor: Alex Ferrer

% Clear workspace, command window and close windows
clear;
close all;
clc;

% Set interpreter to latex
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

%% Brief description
%{

Data from https://www.kaggle.com/c/digit-recognizer/data?select=test.csv

Two files are provided from kaggle: test.csv and train.csv

In this project, both test and train will be done using train.csv renamed
as data.csv

Each row is a number, each column is the pixel number (first column is the label).

%}

%% Computation

% Load data
load('data.mat');

% Rows (m) and columns (n)
m_original = size(data,1);
n_original = size(data,2);

% X and Y data
xdata = data(:,2:end);
ydata = data(:,1);

% Number of data points
nData = m_original;

% Train percentage
percentage = 2.38; % Percentage

% Split data into a train and test set
[Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(xdata,ydata,nData,percentage);

% [W1, b1, W2, b2] = initParameters();
% [Z1, A1, Z2, A2] = forwardPropagation(W1, b1, W2, b2);



[W1, b1, W2, b2] = gradientDescent(Xtrain, Ytrain, 0.10, 500);



% [X,m] = computeFullX(Xtrain,d);








