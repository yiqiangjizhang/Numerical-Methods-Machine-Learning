function [X,m] = computeFullX(Xtrain,d)

%X=[ones(size(Xdata,1),1) Xdata(:,1) Xdata(:,2) Xdata(:,3) Xdata(:,4)];
X1=Xtrain(:,1); 
X2=Xtrain(:,2);
% X3=Xdata(:,3);
% X4=Xdata(:,4);
contador=1;
for g=0:d
    for a=0:g
           X(:,contador)=X2.^(a) .* X1.^(g-a);
           contador=contador+1;
    end
end
               %X4.^(c).*X3.^(b-c).*X2.^(a-b).*X1.^(g-a);%=X4.^(c).*X3.^(b-c).*X2.^(a-b).*X1.^(g-a);
m=contador-1;
end