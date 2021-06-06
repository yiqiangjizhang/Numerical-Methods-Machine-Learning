%-------------------------------------------------------------------------%
% Assignment 2: Linear Regression using Machine Learning
%-------------------------------------------------------------------------%

% Date: 11/05/2021
% Author/s: 
%   Yi Qiang Ji
%   Biel Galiot
%   Gerar Villalta
%   Oriol Miras

% Subject: Robotic Exploration of the Solar System
% Professor: Manel Soria & Arnau Miro

clc;
close all;
clear all;

% Set interpreter to latex
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');


% Loop for 1000 times
iter = 1000;

% Loop for each percentage
p_number = 100; % delta of increment
percentage = linspace(0.01,0.99,p_number); % From [0,1]
MSE = zeros(p_number,iter);

% Loop for every percentage
for j = 1:1:length(percentage)
    
    for i=1:1:iter

        % Number of points
        nData = 200;

        % Generate data
        [xdata,ydata] = generateData(nData);

        % Split train and test data
        [Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(xdata,ydata,nData,percentage(j));

        % Compute coefficients
        w = computeW(Xtrain,Ytrain);

        % Predicted Y
        YtestPredicted = predictor(w,Xtest);

        % Minimum Squared Error
        MSE(j,i) = computeMSE(YtestPredicted,Ytest);

%         % Plots
%         plot(xdata,ydata,'+');
%         hold on
%         plot(Xtest,YtestPredicted,'r+')
    end
    mean_MSE(1,j) = mean(MSE(j,:));
    std_MSE(1,j) = std(MSE(j,:));
end

plot_pdf = figure(1);
errorbar(percentage,mean_MSE(1,:),std_MSE(1,:));
grid on;
grid minor;
box on;
xlabel('Percentage train/test data');
ylabel('Mean $\pm$ $\sigma$');
title('\textbf{Mean Square Error}');

% % Save pdf
% set(plot_pdf, 'Units', 'Centimeters');
% pos = get(plot_pdf, 'Position');
% set(plot_pdf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Centimeters', ...
%     'PaperSize',[pos(3), pos(4)]);
% print(plot_pdf, 'MSE_afi.pdf', '-dpdf', '-r0')
% 
% % Save png
% print(gcf,'MSE_afi.png','-dpng','-r600');

% Functions
function [Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(xdata,ydata,nData,percentage)
    
    % Split data
    % Holdout validation percentage
    h_validation = percentage; % Percentage
    
    train_data = round(h_validation*nData);
    % test_data = nData - train_data;
    
    Xtrain = xdata(1:train_data);
    Xtrain = [xdata(1:train_data) ones(size(Xtrain,1),1)];
    Ytrain = ydata(1:train_data);
    
    Xtest = xdata((train_data+1):end);
    Xtest = [xdata((train_data+1):end) ones(size(Xtest,1),1)];
    Ytest = ydata((train_data+1):end);
end

function [xdata,ydata] = generateData(nData)
    
    % Uniform distributed values between 'init' and 'final'
    % r = a + (b-a).*rand(N,1)
    
    % Initial and final range
    init = -1;
    final = 1;
    
    % Generate data
    xdata = -init + (init-(final))*rand(nData,1);
    ydata = xdata.^2 + 0.01*rand(nData,1); % Add some error
end

function MSE = computeMSE(YtestPredicted,Ytest)
    MSE = immse(Ytest,YtestPredicted);
end

function w = computeW(Xtrain,Ytrain)
    w = (Xtrain.'*Xtrain)\(Xtrain.'*Ytrain);
end

function y = predictor(w,x)
    y = x*w;
end


function y = f(x)
y = -x.^2;
end