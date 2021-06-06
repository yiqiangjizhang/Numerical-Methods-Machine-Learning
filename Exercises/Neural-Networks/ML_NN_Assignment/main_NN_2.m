clc
close all
clear all
%%
iter=50;
plot_bool=false;
fminunc_plot=false;
lambda_min=1e-1;
%Ahora el termino d es dificil porque los temrinos cruzados son muy dificiles 
percentage=25;
d=1;
num_capas_max=5;
figure;
for k=1:5
    lambda(k)=10^(-k);
    
    fprintf("########### LAMBDA: 1e-%d ############ \n",k);
    for i=1:num_capas_max
        num_capas(i)=i+1;
        fprintf("########### NUMBER OF LAYERS: %d ############ \n",i);
        for j=1:iter
            [MSE_train(i,j),MSE_test(i,j)]=LogisticRegression_Neuronal(percentage,d,plot_bool,lambda(k),num_capas(i),fminunc_plot);
        end
    %   median_train(i)=nanmedian(MSE_train(i,:));
    %   std_train(i)=nanstd(MSE_train(i,:));
        median_test(i)=nanmedian(MSE_test(i,:));
        std_test(i)=nanstd(MSE_test(i,:));
    end


    title("Median of the generalization error vs number of layers. Test set to training set ratio: "+ percentage+ "$\%$", 'interpreter','latex')
    xlabel('Layers', 'interpreter','latex');
    ylabel('MSE [\%]', 'interpreter','latex');
    errorbar(num_capas,median_test,std_test,'displayName',sprintf('lambda = %d', lambda(k)));
    grid on;
    box on;
    hold on
    legend('location', 'best', 'interpreter','latex')
    drawnow
end
grid minor;