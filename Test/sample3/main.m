%% Machine Learning
%
%-------------------------------------------------------------------------%
% Car dataset
%
% Read data from car.data and extract it
% 
%-------------------------------------------------------------------------%

% Date: 23/04/2021
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

% Load data

% load cleveland
load flare2
flare2(1,:) = [];

% Separate data
% 
% % age: age in years
% age = processedcleveland(:,1);
% % sex: sex (1 = male; 0 = female)
% sex = processedcleveland(:,2);
% %{
% cp: chest pain type
% -- Value 1: typical angina
% -- Value 2: atypical angina
% -- Value 3: non-anginal pain
% -- Value 4: asymptomatic
% %}
% chest_pain = processedcleveland(:,3);
% % trestbps: resting blood pressure (in mm Hg on admission to the hospital)
% resting_blood_pressure = processedcleveland(:,4);
% % chol: serum cholestoral in mg/dl
% cholesterol = processedcleveland(:,5);
% % fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
% blood_sugar = processedcleveland(:,6);
% % restecg: resting electrocardiographic results
% resting_electrocardiographic_result = processedcleveland(:,7);
% % thalach: maximum heart rate achieved
% max_heart_rate_achieved = processedcleveland(:,8);
% % exang: exercise induced angina (1 = yes; 0 = no)
% exercise_indulced_angina = processedcleveland(:,9);
% % oldpeak = ST depression induced by exercise relative to rest
% ST_depression = processedcleveland(:,10);
% %{
% slope: the slope of the peak exercise ST segment
% -- Value 1: upsloping
% -- Value 2: flat
% -- Value 3: downsloping
% %}
% ST_slope = processedcleveland(:,11);
% % ca: number of major vessels (0-3) colored by flourosopy
% num_vessels = processedcleveland(:,12);
% % thal: 3 = normal; 6 = fixed defect; 7 = reversable defect (duration exercise)
% duration_exercise = processedcleveland(:,13);
% %{
% num: diagnosis of heart disease (angiographic disease status)
% -- Value 0: < 50% diameter narrowing
% -- Value 1: > 50% diameter narrowing
% (in any major vessel: attributes 59 through 68 are vessels)
% %}
% diagnosis  = processedcleveland(:,14);
% 
% heart_disease_table = table(age,sex,chest_pain,resting_blood_pressure,cholesterol, ...
%     blood_sugar, resting_electrocardiographic_result, max_heart_rate_achieved, ...
%     exercise_indulced_angina, ST_depression, ST_slope, num_vessels, duration_exercise);

