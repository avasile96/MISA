function [mask,mu,v,p] = EM(img)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%Define classes
k = 4;

% check image
img = double(img);
copy = img;           % make a copy
img = img(:);         % vectorize ima
% mi = min(img);        % deal with negative, delete line
% img = img-mi+1;       % and zero values, delete line
m = max(img);
s = length(img);

% create image histogram

h=histo(img);
x = find(h); % indices of non-zero elements of h
h = h(x); % we populate h with only its non-zero elements
x = x(:); h = h(:); % from 1x427 becomes 427x1 (traspose)

% initiate parameters

mu=(1:k)*m/(k+1); % initialization of the mean for each class
v=ones(1,k)*m; % init of variances
p=ones(1,k)*1/k; % init of proportions?

% start process

sml = mean(diff(x))/1000; % maybe to avoid dividing by 0 later
while(1)
        % Expectation
        prb = distribution(mu,v,p,x);
        scal = sum(prb,2)+eps;
        loglik=sum(h.*log(scal)); %logarithmic likelihood
        
        %Maximization
        for j=1:k
                pp=h.*prb(:,j)./scal;
                p(j) = sum(pp); % updating proportions
                mu(j) = sum(x.*pp)/p(j); % updating the mean
                vr = (x-mu(j));
                v(j)=sum(vr.*vr.*pp)/p(j)+sml; % updating the mean
        end
        p = p + 1e-3;
        p = p/sum(p);

        % Exit condition
        prb = distribution(mu,v,p,x);
        scal = sum(prb,2)+eps; % scal = smooth histogram of the distribution of gaussians with which we approximate segmentations
        nloglik=sum(h.*log(scal));                
        if((nloglik-loglik)<0.0001), break; end        

        clf
        plot(x,h);
        hold on
        plot(x,prb,'g--')
        plot(x,sum(prb,2),'r')
        drawnow
end

% calculate mask
mu=mu+mi-1;   % recover real range
s=size(copy);
mask=zeros(s);

for i=1:s(1)
for j=1:s(2)
  for n=1:k
    c(n)=distribution(mu(n),v(n),p(n),copy(i,j)); 
  end
  a=find(c==max(c));  
  mask(i,j)=a(1);
end
end
