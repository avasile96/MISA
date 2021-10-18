function [h]=histo(datos)
% datos=datos(:);  % Vectorize the "datos" variable, it can be commented/deleted
% ind=find(isnan(datos)==1);  % Find all the NaN values, it can be commented/deleted
% datos(ind)=0;  % It sets all the NaN values to zero, it can be commented/deleted
% ind=find(isinf(datos)==1);  % Find all the infinite values, it can be commented/deleted
% datos(ind)=0;  % It sets all the infinite values to zero, it can be commented/deleted
tam=length(datos);
m=ceil(max(datos))+1;
h=zeros(1,m);
for i=1:tam  % Loop to get the histogram of the vectorized image
    f=floor(datos(i));    
    if(f>0 && f<(m-1))        
        a2=datos(i)-f;
        a1=1-a2;
        h(f)  =h(f)  + a1;      
        h(f+1)=h(f+1)+ a2;                          
    end
end
% Normalization????
h=conv(h,[1,2,3,2,1]);
h=h(3:(length(h)-2));
h=h/sum(h);
end

