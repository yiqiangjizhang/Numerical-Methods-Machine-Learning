function [MSE_train, MSE_test]=LogisticRegression_Neuronal_Xisco(percentage,d,plot_bool,lambda,num_capas,fminunc_plot)
[Xdata,ydata] = loadData();
nData=length(ydata);

%To plot we have to do a computefullx with d=1;
%X_plot=computeFullX(Xdata,1);
%hold on
%end plot
%v = find_indices(d);
[Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(Xdata,ydata,nData,percentage);

%[m, n] = size(Xtrain); %Esto creo que no lo utilizamos para nada
[X, m] = computeFullX(Xtrain,d);
theta0 = computeInitialTheta(m, num_capas);
theta0_vec = reshape(theta0,[],1);
theta0_mat = reshape(theta0,[m,m,num_capas]);
[cost, grad] = computeCostFunction(X,Ytrain,theta0_vec,lambda,m,num_capas);
[thetaOpt,fval,exitflag,output] = solveTheta(X,Ytrain,theta0_vec,lambda,m,num_capas,fminunc_plot);
MSE_train=fval;
if plot_bool
    figure(1)
    plotBoundary(X,Ytrain,thetaOpt,d,m,num_capas)
end

[mt, nt] = size(Xtest);
[Xt, mt] = computeFullX(Xtest,d);
[cost, grad] = computeCostFunction(Xt, Ytest,thetaOpt,0,m,num_capas);
MSE_test = cost;
end

function [Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(xdata,ydata,nData,percentage)
r=randperm(nData);
ntest=round(percentage/100*nData);
ntrain=nData-round(percentage/100*nData);

Xtrain = xdata(r(1:ntrain),:);
Xtest = xdata(r((ntrain+1):end),:);

Ytrain = ydata(r(1:ntrain),:);
Ytest = ydata(r((ntrain+1):end),:);

end

function [theta,fval,exitflag,output] = solveTheta(X,Ytrain,theta,lambda,m,num_capas,fminunc_plot)
if fminunc_plot
    options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','PlotFcn',{@optimplotfval},'StepTolerance',10^(-9),'MaxFunEvals',5000,'CheckGradients',true);
else
    options = optimoptions(@fminunc,'Algorithm','quasi-newton','StepTolerance',10^(-9),'MaxFunEvals',5000,'CheckGradients',true);
end
F =@(theta) computeCostFunction(X,Ytrain,theta,lambda,m,num_capas);
[theta,fval,exitflag,output] = fminunc(F,theta,options);
end

function plotBoundary(X,y,theta,d,m_new,num_capas)
    n = size(X,2);
    m = size(X,1);
    figure(25)
    plotData(X(:,2),X(:,3),y)
    for j = 1:3
        x1 = linspace(min(X(:,2)),max(X(:,2)),100);
        x1 = x1';
        x2 = min(X(:,3)) + zeros(100,1);
        x2_aux = zeros(100,1);
        m_X = size(X,2);
        X_test = zeros(100,n,100);
        h = zeros(100,100);
        %theta_aux = [theta((j*n-n+1):j*n)];
        theta_mat=reshape(theta,[m_new,m_new,num_capas]);
        %x3 = zeros(100,1);
        %x4 = zeros(100,1);
        %0 = theta(1) + x1*theta(2) + x2*theta(3) + x1^2*theta(4) + x2^2*theta(5)
        for i = 1:100 
            x2 = x2 + (max(X(:,3)) - min(X(:,3)))/100;
            x2_aux(i) = x2(1);
            xdata_test = [x1 , x2];
            X_test(:,:,i) = computeFullX(xdata_test,d);
            [~,z] = foward_propagation(theta_mat, X_test(:,:,i));
            h(:,i) = z(:,j,end);
        end
        
        %contour(x1,x2_aux,h',[0 0])
        hold on

        if j == 1
            contour(x1,x2_aux,h',[0 0],'r')
        elseif j == 2
            contour(x1,x2_aux,h',[0 0],'g')
        elseif j==3
            contour(x1,x2_aux,h',[0 0],'b')
        end
    end
end

function [X,y] = loadData()
data = load('iris.mat');
X = data.meas(:, [3 4]);
ydata = data.meas(:, 5);
y=zeros(length(ydata),max(ydata));
for i=1:length(ydata)
    if ydata(i)==1
        y(i,1)=1;
    elseif ydata(i)==2
        y(i,2)=1;
    else
        y(i,3)=1;
    end
end
end

function plotData(X1,X2,y)
gscatter(X1,X2,y,'bgr','xo*')
xlabel("X3");
ylabel("X4");
end

function [X,m] = computeFullX(Xdata,d)
%X=[ones(size(Xdata,1),1) Xdata(:,1) Xdata(:,2) Xdata(:,3) Xdata(:,4)];
X1=Xdata(:,1); 
X2=Xdata(:,2);
% X3=Xdata(:,3);
% X4=Xdata(:,4);
contador=1;
for g=0:d
    for a=0:g
           X(:,contador)=X2.^(a).*X1.^(g-a);
           contador=contador+1;
    end
end
               %X4.^(c).*X3.^(b-c).*X2.^(a-b).*X1.^(g-a);%=X4.^(c).*X3.^(b-c).*X2.^(a-b).*X1.^(g-a);
m=contador-1;
end

function [X,m] = computeFullX_plot(Xdata,d)
%X=[ones(size(Xdata,1),1) Xdata(:,1) Xdata(:,2) Xdata(:,3) Xdata(:,4)];
X1=Xdata(:,1); 
X2=Xdata(:,2);
contador=1;
for g=0:d
    for a=0:g
           X(:,contador)=X2.^(a).*X1.^(g-a);
           contador=contador+1;
    end
end
m=contador-1;
end

function theta0 = computeInitialTheta(m, num_capas)
theta0=ones(m^2,num_capas);
end

function [J,grad] = computeCostFunction(X,y,theta_vec,lambda, m, num_capas)
o=num_capas;
theta=reshape(theta_vec,[],o);
theta_mat=reshape(theta,[m,m,num_capas]);
[a,z] = foward_propagation(theta_mat, X);
aux= theta_vec'*theta_vec;
a_last=a(:,:,o+1);
for i=1:size(y,2)
J_vec(i) =(1/length(y))*sum((1-y(:,i)).*(-log(1-a_last(:,i)))+y(:,i).*(-log(a_last(:,i))));
end
[delta] = back_propagation(a, theta_mat, z, X, y);
% h = hypothesisFunction(X,theta_mat);
% g = sigmoid(h);
for i=1:o
%grad_vec(:,i)=(1/length(y))*sum((g(:,i)-y(:,i))*X')' + lambda*theta_mat(:,i);
grad_vec(:,:,i)=(1/length(y))*(delta(:,:,i+1)'*a(:,:,i+1))+lambda*theta_mat(:,:,i);
end
%grad_vec=(1/length(y)).*(delta(:,:,o+1)'*a(:,:,o+1));
J=sum(J_vec)+(1/(2*length(y)))*lambda*aux; %aux=(theta_vec'*theta_vec)
grad =reshape(grad_vec,[],1);
end

function [a,z] = foward_propagation(theta,X)
o=size(theta,3); %o=Capas
a=zeros(size(X,1),size(X,2),o+1); 
z=zeros(size(X,1),size(X,2),o+1);
a(:,:,1)=X;
for i=1:o
    z(:,:,i+1)=(theta(:,:,i)*a(:,:,i)')'; %Esto hay que revisarlo, creo que asi esta bien la multilicacion
    a(:,:,i+1)=sigmoid(z(:,:,i+1));
end
end

function [delta] = back_propagation(a, theta, z, X,y)
o=size(theta,3);
delta=zeros(size(X,1),size(X,2),o+1);
delta(:,:,o+1)=a(:,:,o+1)-y;
for i=o:-1:2
    delta(:,:,i)=(theta(:,:,i)*delta(:,:,i+1)')'*dsigmoid(z(:,:,i));
end
end

function h = hypothesisFunction(X,theta)
h=zeros(size(theta,1),size(X,1),size(theta,3));
for i=1:size(theta,3)
h(:,:,i)=theta(:,:,i)*(X)';
end
end

function g = sigmoid(z)
g=1./(1+exp(-(z)));
end
function dg = dsigmoid(z)
dg=sigmoid(z)'*(1-sigmoid(z));
end
