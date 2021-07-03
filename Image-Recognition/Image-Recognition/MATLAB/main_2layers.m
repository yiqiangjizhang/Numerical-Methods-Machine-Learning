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
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaulttextinterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

%% Brief description
%{

Data from https: // www.kaggle.com / c / digit - recognizer / data?select = test.csv

Two files are provided from kaggle:test.csv and train.csv

In this project, both test and train will be done using train.csv renamed
as data.csv

Each row is a number, each column is the pixel number (first column is the label).

%}

%% Computation

% Load data
load('data.mat');

% Rows (m) and columns (n)
m_original = size(data, 1);
n_original = size(data, 2);

% X and Y data
xdata = data(:, 2:end);
ydata = data(:, 1);

% Number of data points
nData = m_original;

% Test percentage
percentatge = 1/42;

% Split data into a train and test set
[Xtrain, Ytrain, Xtest, Ytest] = splitDataTrainAndTest(data, nData, percentatge);

alpha = 0.01;
iterations = 3;
nodes = 10; % Nodes of hidden layers (same number for each layer)

nr_correct = 0; % For accuracy


% gradientDescent(Xtrain, Ytrain, alpha, iterations, nodes, nr_correct, m_original);

% Initiate values
[W1, b1, W2, b2, W1_1, b1_1] = initParameters(Xtrain, nodes);

%{
Gradient Descent
%}

counter = 1;
counter_vec = 0;

for i = 1:1:iterations

    for j = 1:size(Xtrain, 1)
        %[Z1, A1, ~, A2] = forwardPropagation(W1, b1, W2, b2, Xtrain(i,:)');
        % [dW1, db1, dW2, db2] = backwardPropagation(Z1, A1, A2, W2, Xtrain, Ytrain, j, W1, b1, b2, alpha);
        %[W1, b1, W2, b2] = updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha);

        %{
        Forward propagation
        %}
        Z1 = W1 * Xtrain(j, :)' + b1;
        % A1 = ReLU(Z1);
        % A1 = Sigmoid(Z1);
        
        Z1_1 = W1_1 * A1 + b1_1;
        % A1_1 = ReLU(Z1_1);
        A1_1 = Sigmoid(Z1_1);
        
        Z2 = W2' * A1_1 + b2;
        % A2 = softmax(Z2);
        A2 = Sigmoid(Z2);

        % Cost / Error
        m = length(Ytrain);
        one_hot_Y = onehot(Ytrain(j));
        error_MSE = 1 / m * sum((A2 - one_hot_Y).^2); % MSE
        
        
        [~, A2_index] = max(A2);
        [~, onehot_index] = max(one_hot_Y);

        nr_correct = nr_correct + (A2_index == onehot_index);

        %{
        Backward propagation
        %}

        delta_o = A2 - one_hot_Y;

        W2 = W2 - alpha * delta_o * A1_1';
        b2 = b2 - alpha * delta_o;
        
        delta_h1 = W2' * (delta_o .* Sigmoid_deriv(A1_1));
        
        W1_1 = W1_1 - alpha * delta_o * A1_1';
        b1_1 = b1_1 - alpha * delta_o;

        delta_h = W1_1' * (delta_h1 .* Sigmoid_deriv(A1));

        % W1 = W2 - alpha * delta_h * Xtrain(j, :)';
        W1 = W1 - alpha * delta_h * Xtrain(j, :);
        b1 = b1 - alpha * delta_h;

        %{
        Accuracy
        %}

    end
    mean_error_MSE(i) = mean(error_MSE);

    if mod(i, 25) == 0
        accuracy(counter) = (nr_correct / m_original) * 100;
        counter_vec(counter) = i;
        fprintf('Accuracy: %f \n', accuracy);
        counter = counter + 1;
    end

    nr_correct = 0;
    

end

% Plot MSE
figure
plot(mean_error_MSE);
box on
grid on
grid minor
curtick = get(gca, 'xTick');
xticks(unique(round(curtick)));

xlabel("$Layer$");
ylabel("$MSE$");


% Plot accuracy
figure
plot(counter_vec, accuracy);
box on
grid on
grid minor
curtick = get(gca, 'xTick');
xticks(unique(round(curtick)));

xlabel("$Epoch$");
ylabel("$Accuracy$");







