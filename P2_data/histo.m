function [h]=histo(datos)
% datos=datos(:);  % Vectorize the "datos" variable, it can be commented/deleted
% ind=find(isnan(datos)==1);  % Find all the NaN values, it can be commented/deleted
% datos(ind)=0;  % It sets all the NaN values to zero, it can be commented/deleted
% ind=find(isinf(datos)==1);  % Find all the infinite values, it can be commented/deleted
% datos(ind)=0;  % It sets all the infinite values to zero, it can be commented/deleted
tam=length(datos);
m=ceil(max(datos))+1;
m = max(m); % we work with the larger histogram vector 
h=zeros(m,2);
for j=1:size(datos,2) % multichannel implementation
    for i=1:tam  % Loop to get the histogram of the vectorized image
        f=floor(datos(i,j));    
%         if(f>0 && f<(m-1))
        if and(f>0,f<(m-1))
            a2=datos(i,j)-f;
            a1=1-a2;
            h(f,j)  =h(f)  + a1;      
            h(f+1,j)=h(f+1)+ a2;                          
        end
    end
end
% Normalization????
for i=1:2
    h(:,i)=conv(h(:,i),[1,2,3,2,1]);    
end
h=h(3:(length(h)-2)); % to get rid of the extra values introduced in the convolution
h=h/sum(h);
end

