clc
close all
clear all
%%
iter=10;
plot_bool=false;
fminunc_plot=false;
lambda=1e-2;
d=1;
num_capas=3;
figure;
for k=1:3
    percentage(k)=25*k;
    
    fprintf("########### PERCENTAGE: %d % ############ \n",percentage(k));
    for i=1:6
        lambda(i)=10^(-i);
        fprintf("########### LAMBDA: 1e-%d ############ \n",i);
        for j=1:iter
            [MSE_train(i,j),MSE_test(i,j)]=LogisticRegression_Neuronal(percentage(k),d,plot_bool,lambda(i),num_capas,fminunc_plot);
        end
    %   median_train(i)=nanmedian(MSE_train(i,:));
    %   std_train(i)=nanstd(MSE_train(i,:));
        median_test(i)=median(MSE_test(i,:));
        std_test(i)=std(MSE_test(i,:));
    end


    title("Median of the generalization error vs lambda. Number of layers: "+ num_capas , 'interpreter','latex')
    xlabel('Lambda', 'interpreter','latex');
    ylabel('MSE [\%]', 'interpreter','latex');
    errorbar(lambda,median_test,std_test,'displayName',sprintf('Percentage = %d', percentage(k)));
    set(gca,'xscale','log');
    grid on;
    box on;
    hold on
    legend('location', 'best', 'interpreter','latex')
    drawnow
end
grid minor;