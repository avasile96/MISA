%Clear workspace, close all windows, clear command window
clear; close all;clc;

%Start time
tic;

label = load_untouch_nii('D:\Uni\Spain\MISA\MISA\P2_data\1\LabelsForTesting.nii');
label_img = label.img(:,:,24);  % Image for the dice coefficient, with labels
BW = imbinarize(double(label_img));  % To obtain the mask to get rid of the skull

img_T1 = load_untouch_nii('D:\Uni\Spain\MISA\MISA\P2_data\1\T1.nii');
img_T2 = load_untouch_nii('D:\Uni\Spain\MISA\MISA\P2_data\1\T2_FLAIR.nii');
Img_T1 = img_T1.img(:,:,24);
Img_T2 = img_T2.img(:,:,24);
%[mask,mu,v,p]=EMSeg(Img,5);  % Original function from link

IMG_T1 = double(Img_T1).* BW;  % Multiplication so we get rid of the skull
IMG_T2 = double(Img_T2).* BW;  % Multiplication so we get rid of the skull

[mask,mu,v,p]=EM(IMG_T1, IMG_T2);  % Customized function
mask = mask-1; %temporary ;) correction for #of clusters = 4
similarity = dice(mask,double(label_img));
disp(similarity);

figure;
subplot(2,2,1); imshow(IMG,[]), title('Original image');
subplot(2,2,2); imshow(label_img,[]), title('Image with its labels');
subplot(2,2,3); imshow(mask,[]), title('Result');


%Finish time and display it
toc;
