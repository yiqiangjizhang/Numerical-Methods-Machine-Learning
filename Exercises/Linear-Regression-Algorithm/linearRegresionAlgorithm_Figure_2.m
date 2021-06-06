
clc;
close all;
clear all;

% Loop for 1000 times
iter = 1000;

% Loop for each nData
data = 10:5:500; % From [10,500]
MSE = zeros(length(data),iter);

% Loop for every nData
for j = 1:1:length(data)
    
    for i=1:1:iter

        % Number of points
        nData = data(j);

        % Generate data
        [xdata,ydata] = generateData(nData);

        % Split train and test data
        [Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(xdata,ydata,nData);

        % Compute coefficients
        w = computeW(Xtrain,Ytrain);

        % Predicted Y
        YtestPredicted = predictor(w,Xtest);

        % Minimum Squared Error
        MSE(j,i) = computeMSE(YtestPredicted,Ytest);

    end
    mean_MSE(1,j) = mean(MSE(j,:));
    std_MSE(1,j) = std(MSE(j,:));
end

plot_pdf = figure(1);
errorbar(data,mean_MSE(1,:),std_MSE(1,:));
grid on;
grid minor;
box on;
xlabel('nData');
ylabel('Mean $\pm$ $\sigma$');
title('\textbf{Mean Square Error}');

% % Save pdf
% set(plot_pdf, 'Units', 'Centimeters');
% pos = get(plot_pdf, 'Position');
% set(plot_pdf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Centimeters', ...
%     'PaperSize',[pos(3), pos(4)]);
% print(plot_pdf, 'MSE_error_ndata_25.pdf', '-dpdf', '-r0')
% 
% % Save png
% print(gcf,'MSE_error_ndata_25.png','-dpng','-r600');

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