function testDiffusion ()
num_iter = 100;
delta_t = 0.25;
kappa = 5;
option = 1;

% im1 = rgb2gray(imread('pa6-16_t2.png'));
nii = load_nii('D:\USB\LAB1\l1_preprocessing\braindata\braindata\t1_icbm_normal_1mm_pn0_rf0.nii');
im1 = double(nii.img(:,:,70));
%im1 = (dicomread('slice_3D.dcm'));

%blurring
H = fspecial('disk',3);
blurred = imfilter(im1,H,'replicate');

ad = anisodiff(im1,num_iter,kappa,delta_t,option);
figure, subplot 131, imshow(im1,[]), subplot 132, imshow(ad,[]), subplot 133, imshow(blurred,[])



