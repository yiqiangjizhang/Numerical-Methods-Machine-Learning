
function MSE=LogisticRegressionOrder_d(percentage,d,plot_bool,lambda)
[Xdata,ydata] = loadData();
nData=length(ydata);

[Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(Xdata,ydata,nData,percentage);

figure(1)
plotData(Xdata, ydata);

[m, n] = size(Xtrain);
X = computeFullX(Xtrain);
theta0 = computeInitialTheta(n,d);
[cost, grad] = computeCostFunction(X, Ytrain,theta0,d,lambda);
[thetaOpt,fval,exitflag,output] = solveTheta(X,Ytrain,theta0,d,lambda);
if plot_bool
    figure(1)
    plotBoundary(X,ydata,thetaOpt,d)
end

[mt, nt] = size(Xtest);
Xt = computeFullX(Xtest);
[cost, grad] = computeCostFunction(Xt, Ytest,thetaOpt,d,0);
MSE = cost;

end


function [Xtrain,Ytrain,Xtest,Ytest] = splitDataTrainAndTest(xdata,ydata,nData,percentage)
r=randperm(nData);
ntest=round(percentage/100*nData);
ntrain=nData-round(percentage/100*nData);

Xtrain = xdata(r(1:ntrain),:);
Xtest = xdata(r((ntrain+1):end),:);

Ytrain = ydata(r(1:ntrain));
Ytest = ydata(r((ntrain+1):end));

end


function [theta,fval,exitflag,output] = solveTheta(X,y,theta0,d,lambda)
options = optimoptions(@fminunc,'PlotFcn','optimplotfval','Algorithm','quasi-newton');
F =@(theta) computeCostFunction(X,y,theta,d,lambda);
[theta,fval,exitflag,output] = fminunc(F,theta0,options);
end

function plotBoundary(X,y,theta,d)
sampling=1000;
x_rec1=linspace(min(X(1,:)),max(X(1,:)),sampling);
x_rec2=linspace(min(X(2,:)),max(X(2,:)),sampling);
[X,Y] = meshgrid(x_rec1,x_rec2);
h=0;
for i=1:d
    h=theta(1,i)*X.^i+theta(2,i)*Y.^i+ones(sampling,sampling)*theta(3,i)+h;
end
hold on
v = [0,0];
contour(X,Y,h,v,'k')

end


function [X,y] = loadData()
data = load('microchip.txt');
X = data(:, [1, 2]); 
y = data(:, 3);
end

function plotData(X,y)
gscatter(X(:,1),X(:,2),y,'rb','xo')
xlabel("Microchip test 1");
ylabel("Microchip test 2");
end


function X = computeFullX(Xdata)
X=[Xdata(:,1),Xdata(:,2),ones(size(Xdata,1),1)]';
end

function theta0 = computeInitialTheta(n,d)
theta0=zeros(n+1,d);
end

function [J,grad] = computeCostFunction(X,y,theta,d,lambda)
h = hypothesisFunction(X,theta,d);
g = sigmoid(h);
fullTheta=reshape(theta,[],1); % convert matrix to column vector

J = (1/length(y))*sum((1-y).*(-log10(1-g'))+y.*(-log10(g'))) + 0.5*lambda*fullTheta'*fullTheta ;
grad=(1/length(y))*sum((g'-y).*X') + lambda*fullTheta;
end

function h = hypothesisFunction(X,theta,d)
h=0;
for i=1:d
    h=h+theta(:,i)'*(X.^i);
end
end

function g = sigmoid(z)
g=1./(1+exp(-(z)));
end