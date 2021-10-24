function [mask,mu,v,p] = EM(IMG_T1, IMG_T2)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%Define classes
k = 4;

% check image
img_T1 = double(IMG_T1);
img_T2 = double(IMG_T2);
copy_T1 = img_T1;           % make a copy
copy_T2 = img_T2;           % make a copy
% img_T1T2 = normalize([img_T1(:) img_T2(:)],'range');      % vectorize ima 
img_T1T2 = [img_T1(:) img_T2(:)];% vectorize ima 
mi_T1T2 = min(img_T1T2);        % deal with negative, delete line
img_T1T2 = img_T1T2-mi_T1T2+1;       % and zero values, delete line
m_T1T2 = max(img_T1T2);
s = length(img_T1T2);

% create image histogram

% h=histo(img_T1T2);
h = histcounts2(img_T1T2(:,1),img_T1T2(:,2),m_T1T2, 'Normalization','probability', 'BinMethod','integers');
% sum(sum(h))
figure, imshow(h(2:end,2:end),[])
h = imgaussfilt(h,0.37);
sum(sum(h))
figure, imshow(h(2:end,2:end),[])

[x, y] = find(h); % indices of non-zero elements of h % just the GSV of the image
h1 = h(x,y); % we populate h with only its non-zero elements
x = x';
y = y';
h = h'; % from 1x427 becomes 427x1 (traspose)

% initiate parameters
k_2D = [1:k; 1:k]';
mu=k_2D.*m_T1T2./(k+1); % initialization of the mean for each class
v=ones(2,k)'.*m_T1T2; % init of variances
p=ones(2,k)'.*1./k; % init of proportions

% start process

sml = [mean(diff(x)), mean(diff(y))]./1000; % quantity to update by
while(1)
        % Expectation
        prb = distribution(mu,v,p,x,y);
        scal = sum(prb,3)+eps;
        loglik=sum(h.*log(scal)); %logarithmic likelihood
        
        %Maximization
        for j=1:k
                pp=h.*prb(j,:)./scal;
                p(j) = sum(pp(:)); % updating proportions
                mu(j,:) = sum(x.*pp(:))/p(j,:); % updating the mean
                vr = (x-mu(j));
                v(j,:)=sum(vr.*vr.*pp)/p(j)+sml; % updating the variance
        end
        p = p + 1e-3;
        p = p/sum(p);

        % Exit condition
        prb = distribution(mu,v,p,x);
        scal = sum(prb,2)+eps; % scal = smooth histogram of the distribution of gaussians with which we approximate segmentations
        nloglik=sum(h.*log(scal));                
        if((nloglik-loglik)<0.0001), break; end        

        clf
        plot(x,h,'DisplayName','original image histo');
        hold on
        plot(x,prb,'g--','DisplayName','aproximated distribution')
        hold on
        plot(x,sum(prb,2),'r','DisplayName','convolved distributions')
        legend
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
