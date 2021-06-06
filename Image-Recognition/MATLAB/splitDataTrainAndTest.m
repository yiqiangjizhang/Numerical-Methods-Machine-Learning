function [Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(xdata,ydata,nData,percentage)
%{
This function splits the set of data into training and test
%}


% METHOD 1: Percentages

% % Random permutation, returns avector with random permutation of nData values
% r=randperm(nData);
% 
% % Train data
% nTrain=round(percentage/100*nData);
% 
% % Test data
% nTest=nData-nTrain;
% 
% % Train data
% Xtrain = xdata(r(1:nTrain),:)';
% Ytrain = ydata(r(1:nTrain),:)';
% 
% % Test data
% Xtest = xdata(r((nTrain+1):end),:)';
% Ytest = ydata(r((nTrain+1):end),:)';

% METHOD 2: NUMBER 
% Train data
nTrain=1000;

% Test data
nTest=nData-nTrain;

% Train data
Xtrain = xdata(1:nTrain,:)';
Ytrain = ydata(1:nTrain,:)';

% Test data
Xtest = xdata(nTrain+1:end,:)';
Ytest = ydata(nTrain+1:end,:)';

end