%% Machine Learning
%
%-------------------------------------------------------------------------%
% Airfoil Self-Noise Data Set
%
% NASA data set, obtained from a series of aerodynamic and acoustic tests of two and three-dimensional airfoil blade sections conducted in an anechoic wind tunnel.
% URL: https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise#
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
load airfoilselfnoise

% Read data
% 1. Frequency, in Hertzs.
frequency = airfoilselfnoise(:,1);
% 2. Angle of attack, in degrees.
angle_of_attack = airfoilselfnoise(:,2);
% 3. Chord length, in meters.
chord_length = airfoilselfnoise(:,3);
% 4. Free-stream velocity, in meters per second.
free_stream_velocity = airfoilselfnoise(:,4);
% 5. Suction side displacement thickness, in meters.
suction_side_displ = airfoilselfnoise(:,5);
% 6. Scaled sound pressure level, in decibels.
scaled_sound_pressure = airfoilselfnoise(:,6);


airfoil_self_noise = table(frequency, angle_of_attack, chord_length, ...
    free_stream_velocity, suction_side_displ, scaled_sound_pressure);

