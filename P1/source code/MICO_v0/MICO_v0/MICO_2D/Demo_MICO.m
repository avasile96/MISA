% This Matlab file demomstrates the method for simultaneous segmentation and bias field correction
% in Chunming Li et al's paper:
%    "Multiplicative intrinsic component optimization (MICO) for MRI bias field estimation and tissue segmentation",
%     Magnetic Resonance Imaging, vol. 32 (7), pp. 913-923, 2014
% Author: Chunming Li, all rights reserved
% E-mail: li_chunming@hotmail.com
% URL:  http://imagecomputing.org/~cmli/

clc;close all;clear all;
iterNum = 10;
N_region=3; % Number of regions % Poor code etiquete, this is hard coded below (basically every time you see a 3)
q=1; % Fuzzifier
%Img=imread('brainweb64.tif');
% Img=imread('C:\Users\robert\Documents\docencia\Udg\MIA_vibot\labs\2017\pre-processing\data\pa4-16_t2.png');


Img=imread('brainweb64.tif');
Img = double(Img(:,:,1));
%load ROI
A=255;
Img_original = Img;
[nrow,ncol] = size(Img); % getting the image dimensions
n = nrow*ncol; % # of px

ROI = (Img>20); ROI = double(ROI); % Simple Thresholding (selecting GSV>20)

tic

Bas=getBasisOrder3(nrow,ncol); % Getting the set of Basis functions used
% to approximate the Bias Field later on % 3rd order functions
N_bas=size(Bas,3);
% We calculate product permutations of different Bias Basis function
% in the region of interest
for ii=1:N_bas
    ImgG{ii} = Img.*Bas(:,:,ii).*ROI;
    for jj=ii:N_bas
        GGT{ii,jj} = Bas(:,:,ii).*Bas(:,:,jj).*ROI;
        GGT{jj,ii} = GGT{ii,jj} ;
    end
end 


energy_MICO = zeros(3,iterNum); % Energy Matrix

b=ones(size(Img)); % Bias Field initialization?
for ini_num = 1:1
    C=rand(3,1); % types of tissues represented by constants
    C=C*A; % Getting 3 random numbers from 0 to 255
    M=rand(nrow,ncol,3); % field of membership functions' random initialization
    a=sum(M,3);
    for k = 1 : N_region
        M(:,:,k)=M(:,:,k)./a; % Normalization step
    end
    
    % We get the index of the maximum elemnt of M along the 3rd dimension
    [e_max,N_max] = max(M,[], 3); 
    % from 1 to size of the depth of M (number of tissues really)
    for kk=1:size(M,3)
        % The tissue plane of M gets to be one if the index of the maximum
        % is the actual tissue of the current iterration i.e. we sort
        % tissues in order of GSV really
        M(:,:,kk) = (N_max == kk); 
    end
    
    M_old = M; chg=10000;
    energy_MICO(ini_num,1) = get_energy(Img,b,C,M,ROI,q);
    
    
    for n = 2:iterNum
        pause(0.1)
        
        [M, b, C]=  MICO(Img,q,ROI,M,C,b,Bas,GGT,ImgG,1, 1);
        energy_MICO(ini_num,n) = get_energy(Img,b,C,M,ROI,q); % Variable 
        % used for storing the values of energy after each iterration
        
        figure(2),
        if(mod(n,1) == 0)
            PC=zeros(size(Img));
            for k = 1 : N_region
                PC=PC+C(k)*M(:,:,k);
            end
            subplot(241),imshow(uint8(Img)),title('original')
            subplot(242),imshow(PC.*ROI,[]); colormap(gray);
            iterNums=['segmentation: ',num2str(n), ' iterations'];
            title(iterNums);
            subplot(243),imshow(b.*ROI,[]),title('bias field')
            img_bc = Img./b;  % bias field corrected image
            subplot(244),imshow(uint8(img_bc.*ROI),[]),title('bias corrected')
            subplot(2,4,[5 6 7 8]),plot(energy_MICO(ini_num,:))
            xlabel('iteration number');
            ylabel('energy');
            pause(0.1)
        end
    end
end

[M,C]=sortMemC(M,C);
seg=zeros(size(Img));
for k = 1 : N_region
    seg=seg+k*M(:,:,k);   % label the k-th region 
end
figure;
subplot(141),imshow(Img,[]),title('Original image');
subplot(142),imshow(seg.*ROI,[]),title('Segmentation result');
subplot(143),imshow(b.*ROI,[]),title('bias field')
subplot(144),imshow(uint8(img_bc.*ROI),[]),title('bias corrected')



