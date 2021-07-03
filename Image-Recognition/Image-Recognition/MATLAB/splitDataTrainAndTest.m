function [Xtrain, Ytrain, Xtest, Ytest] = splitDataTrainAndTest(data, nData, percentatge)
    %{
    This function splits the set of data into training and test
    %}

    nTest = percentatge * size(data, 1);

    % Test data

    random_shuffle = data(randperm(size(data, 1)), :);

    % Test data
    nTrain = nData - nTest;

    % Train data
    Xtrain = random_shuffle(1:nTrain, 2:end) ./ 255; % To normalize
    Ytrain = random_shuffle(1:nTrain, 1);

    % Test data
    Xtest = random_shuffle(nTrain + 1:end, 2:end) ./ 255; % To normalize
    Ytest = random_shuffle(nTrain + 1:end, 1);

end
