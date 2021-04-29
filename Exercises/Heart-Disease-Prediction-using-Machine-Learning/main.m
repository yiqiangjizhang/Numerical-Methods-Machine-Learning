%% Machine Learning
%
%-------------------------------------------------------------------------%
% Heart Disease Data Set
%
% This database contains 76 attributes, but all published experiments refer to using 
% a subset of 14 of them. In particular, the Cleveland database is the only one that 
% has been used by ML researchers to this date. 
%
% The "goal" field refers to the presence of heart disease in the patient. 
% It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland 
% database have concentrated on simply attempting to distinguish presence 
% (values 1,2,3,4) from absence (value 0).
%
% https://archive.ics.uci.edu/ml/datasets/Heart+Disease
% url: https://www.kaggle.com/ronitf/heart-disease-uci
%-------------------------------------------------------------------------%

% Date: 29/04/2021
% Author/s: Yi Qiang Ji Zhang
% Subject: NUMERICAL TOOLS IN MACHINE LEARNING FOR AERONAUTICAL ENGINEERING
% Professor: Alex Ferrer Ferre

% Clear workspace, command window and close windows
clear;
close all;
clc;

% Set interpreter to latex
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

%% Analyse data

fine_tree = 73.3;
medium_tree = 73.3;
coarse_tree = 80.0;
linear_discriminant = 82.7;
quadratic_discriminant = 81.3;
logistic_regression = 84.0;
naive_bayes_gaussian = 86.7;
naive_bayes_kernel = 80.0;
SVM_linear = 84.0;
SVM_quadratic = 85.3;
SVM_cubic = 72.0;
SVM_fine_gaussian = 58.7;
SVM_medium_gaussian = 84.0;
SVM_coarse_gaussian = 82.7;
KNN_fine = 76.0;
KNN_medium = 85.3;
KNN_coarse = 84.0;
KNN_cosine = 82.7;
KNN_cuibc = 81.3;
KNN_weighted = 86.7;
Ensemble_boosted_trees = 78.7;
Ensemble_bagged_trees = 86.7;
Ensemble_subspace_discriminant = 84.0;
Ensemble_subspace_KNN = 70.7;
Ensemble_RUSBoosted_trees = 81.3;

x =  categorical({'fine tree', 'medium tree', 'coarse tree', 'linear discriminant', ...
    'quadratic discriminant', 'logistic regression', 'naive bayes gaussian', ...
    'naive bayes kernel', 'SVM linear', 'SVM quadratic', 'SVM cubic', ...
    'SVM fine gaussian', 'SVM medium gaussian', 'SVM coarse gaussian', ...
    'KNN fine', 'KNN medium', 'KNN coarse', 'KNN cosine', 'KNN cuibc', ...
    'KNN weighted', 'Ensemble boosted trees', 'Ensemble bagged trees', ...
    'Ensemble subspace discriminant', 'Ensemble subspace KNN', ...
    'Ensemble RUSBoosted trees'});

y = [fine_tree;
medium_tree;
coarse_tree;
linear_discriminant;
quadratic_discriminant;
logistic_regression;
naive_bayes_gaussian;
naive_bayes_kernel;
SVM_linear;
SVM_quadratic;
SVM_cubic;
SVM_fine_gaussian;
SVM_medium_gaussian;
SVM_coarse_gaussian;
KNN_fine;
KNN_medium;
KNN_coarse;
KNN_cosine;
KNN_cuibc;
KNN_weighted;
Ensemble_boosted_trees;
Ensemble_bagged_trees;
Ensemble_subspace_discriminant;
Ensemble_subspace_KNN;
Ensemble_RUSBoosted_trees]';

% Plot
plot1 = figure(1);
bar(x,y)
% text(1:length(y),y,num2str(y'),'HorizontalAlignment','center','VerticalAlignment','bottom'); 
set(plot1,'Position',[475 150 1000 800])
ylabel('Accuracy (%)')
title('Plot')
box on
grid minor
hold off;

% % Save pdf
% set(plot1, 'Units', 'Centimeters');
% pos = get(plot1, 'Position');
% set(plot1, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Centimeters', ...
%     'PaperSize',[pos(3), pos(4)]);
% print(plot1, 'accuracy.pdf', '-dpdf', '-r0');
% 
% % Save png
% print(gcf,'accuracy.png','-dpng','-r100');



















