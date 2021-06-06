clc
close all
clear all
%%
iter=1;
plot_bool=true;
fminunc_plot=true;
lambda=1e-3;
num_capas=3;
percentage=25;
d=1;
[MSE_train,MSE_test]=LogisticRegression_Neuronal(percentage,d,plot_bool,lambda,num_capas,fminunc_plot);