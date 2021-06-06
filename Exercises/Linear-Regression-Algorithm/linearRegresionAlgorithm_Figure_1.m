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


% Loop for every iteration
iter = 1000;
MSE = zeros(1,iter);

for i=1:1:iter
    % Number of points
    nData = 200;

    % Generate data
    [xdata,ydata] = generateData(nData);

    % Split train and test data
    [Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(xdata,ydata,nData);

    % Compute coefficients
    w = computeW(Xtrain,Ytrain);

    % Predicted Y
    YtestPredicted = predictor(w,Xtest);

    % Minimum Squared Error
    MSE(i) = computeMSE(YtestPredicted,Ytest);

%     % Plots
%     figure(1)
%     plot(xdata,ydata,'+');
%     hold on
%     plot(Xtest,YtestPredicted,'r+')

end

% plot_pdf = figure(1);
% grid on;
% grid minor;
% box on;
% xlabel('x');
% ylabel('y');
% title('\textbf{Real data vs Predicted model}');
% legend('Real data', 'Predicted model');

% % Save pdf
% set(plot_pdf, 'Units', 'Centimeters');
% pos = get(plot_pdf, 'Position');
% set(plot_pdf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Centimeters', ...
%     'PaperSize',[pos(3), pos(4)]);
% print(plot_pdf, 'data_vs_predicted.pdf', '-dpdf', '-r0')

% Save png
% print(gcf,'data_vs_predicted.png','-dpng','-r600');

plot_pdf2 = figure(2);
histogram(MSE);
grid on;
grid minor;
box on;
xlabel('MSE');
ylabel('Iterations');
title('\textbf{Mean Square Error (TEST = 25\%)}');

% % Save pdf
% set(plot_pdf2, 'Units', 'Centimeters');
% pos = get(plot_pdf2, 'Position');
% set(plot_pdf2, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Centimeters', ...
%     'PaperSize',[pos(3), pos(4)]);
% print(plot_pdf2, 'MSE_25_test.pdf', '-dpdf', '-r0')
% 
% % Save png
% print(gcf,'MSE_25_test.png','-dpng','-r600');

% Functions
function [Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(xdata,ydata,nData)
    
    % Split data
    % Holdout validation percentage
    h_validation = 0.25; % Percentage
    
    train_data = round(h_validation*nData);
    % test_data = nData - train_data;
    
    Xtrain = xdata(1:train_data);
    Ytrain = ydata(1:train_data);
    Xtest = xdata((train_data+1):end);
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
    y = w*x;
end

function y = f(x)
y = -x.^2;
end