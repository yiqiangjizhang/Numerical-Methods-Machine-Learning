clc
close all
clear all
%%
iter=10;
plot_bool=false;
lambda=0;
for m=1:3
    percentage(m)=25*m;
    for j=1:7
        d(j)=j+1;
        for i=1:iter
            if i==iter
                plot_bool=true;
            else
                plot_bool=false;
            end
            MSE(i,j)=LogisticRegressionOrder_d(percentage(m),d(j),plot_bool,lambda);
        end
        median_val(j)=median(MSE(:,j));
        std_val(j)=std(MSE(:,j));
    end

    figure;
    errorbar(d,median_val,std_val)
    title('Median of the MSE for each polynamial order $d$. Test set to training set ratio:'+ percentage(m)+ '$\%$', 'interpreter','latex')
    xlabel('Polynamial order $d$', 'interpreter','latex');
    ylabel('MSE [%]', 'interpreter','latex');
    grid on;
    grid minor;
    box on;
end