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


% Divisions
k_div = 2:1:50;

% MSE vector
MSE = zeros(1,length(k_div));

% Number of points
nData = 200;

% Generate data
[xdata,ydata] = generateData(nData);


for j=1:1:length(k_div)
    % Loop for every division
    for i=1:1:k_div(j)
        
        MSE_div = zeros(k_div(j),1);
        % Split train and test data
        [Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(xdata,ydata,nData, i, k_div(j));

        % Compute coefficients
        w = computeW(Xtrain,Ytrain);

        % Predicted Y
        YtestPredicted = predictor(w,Xtest);

        % Minimum Squared Error
        MSE_div(i,:) = computeMSE(YtestPredicted,Ytest);
    %     % Plots
    %     figure(1)
    %     plot(xdata,ydata,'+');
    %     hold on
    %     plot(Xtest,YtestPredicted,'r+')
    end
        MSE(1,j) = mean(MSE_div);
        mean_MSE = MSE;
        std_MSE(1,j) = std(MSE_div);
        hold on;
end

plot_pdf = figure(1);
histogram(MSE);
grid on;
grid minor;
box on;
xlabel('MSE');
ylabel('Iterations');
title('\textbf{Mean Square Error using k-fold cross validation}');

% % Save pdf
% set(plot_pdf, 'Units', 'Centimeters');
% pos = get(plot_pdf, 'Position');
% set(plot_pdf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Centimeters', ...
%     'PaperSize',[pos(3), pos(4)]);
% print(plot_pdf, 'MSE_afi_kfold.pdf', '-dpdf', '-r0')
% 
% % Save png
% print(gcf,'MSE_afi_kfold.png','-dpng','-r600');

plot_pdf2 = figure(2);
errorbar(k_div,mean_MSE(1,:),std_MSE(1,:));
grid on;
grid minor;
box on;
xlabel('Number of $k$ divisions');
ylabel('Mean $\pm$ $\sigma$');
title('\textbf{Mean Square Error}');

% % Save pdf
% set(plot_pdf2, 'Units', 'Centimeters');
% pos = get(plot_pdf2, 'Position');
% set(plot_pdf2, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Centimeters', ...
%     'PaperSize',[pos(3), pos(4)]);
% print(plot_pdf2, 'MSE_afi_kfold_error.pdf', '-dpdf', '-r0')
% 
% % Save png
% print(gcf,'MSE_afi_kfold_error.png','-dpng','-r600');

% Functions
function [Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(xdata,ydata,nData, index, k_div)
    
    % Split data
    % Holdout validation percentage
    % h_validation = 0.25; % Percentage
    
    % train_data = round(h_validation*nData);
    % test_data = nData - train_data;
    
    div = floor(nData/k_div);
    Xtest = xdata(((index-1)*div + 1):index*div);
    Xtest = [xdata(((index-1)*div + 1):index*div) ones(size(Xtest,1),1)];
    Ytest = ydata(((index-1)*div + 1):index*div);
    
    Xtrain = setdiff(xdata,Xtest);
    Xtrain = [setdiff(xdata,Xtest) ones(size(Xtrain,1),1)];
    Ytrain = setdiff(ydata,Ytest);
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