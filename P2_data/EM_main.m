%Clear workspace, close all windows, clear command window
clear; close all;clc;

%Start time
tic;

label = load_untouch_nii('F:\DISCO_DURO\Mixto\Subjects\GitHub\MISA\P2_data\1\LabelsForTesting.nii');
label_img = label.img(:,:,24);  % Image for the dice coefficient, with labels
BW = imbinarize(double(label_img));  % To obtain the mask to get rid of the skull

img = load_untouch_nii('F:\DISCO_DURO\Mixto\Subjects\GitHub\MISA\P2_data\1\T1.nii');
Img = img.img(:,:,24);
%[mask,mu,v,p]=EMSeg(Img,5);  % Original function from link

IMG = double(Img).* BW;  % Multiplication so we get rid of the skull

[mask,mu,v,p]=EM(IMG);  % Customized function

similarity = dice(mask,double(label_img));
disp(similarity);

figure;
subplot(2,2,1); imshow(IMG,[]), title('Original image');
subplot(2,2,2); imshow(label_img,[]), title('Image with its labels');
subplot(2,2,3); imshow(mask,[]), title('Next');
subplot(2,2,4); imshow(mask,[]), title('Next'); %Image missing,  thinking which image can be used

%Finish time and display it
toc;
