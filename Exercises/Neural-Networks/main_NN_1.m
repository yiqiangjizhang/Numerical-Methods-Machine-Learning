clc
close all
clear all
%%
iter=1;
plot_bool=true;
lambda=1e-3;
%Ahora el termino d es dificil porque los temrinos cruzados son muy dificiles 
percentage=25;
d=1;
[MSE_train,MSE_test]=LogisticRegression_Neuronal(percentage,d,plot_bool,lambda);

%             median_train(i)=nanmedian(MSE_train(i,:));
%             std_train(i)=nanstd(MSE_train(i,:));
%             median_test(i)=nanmedian(MSE_test(i,:));
%             std_test(i)=nanstd(MSE_test(i,:));

%     figure;
%     errorbar(iterations,median_train,std_train)
%     title("Median of the MSE vs iterations for $\lambda = $'"+lambda+". Test set to training set ratio: "+ percentage+ "$\%$", 'interpreter','latex')
%     xlabel('Iterations', 'interpreter','latex');
%     ylabel('MSE [\%]', 'interpreter','latex');
%     grid on;
%     grid minor;
%     box on;
%     hold on
%     errorbar(iterations,median_test,std_test);
%     legend('Training error', 'Generalization error', 'interpreter','latex')