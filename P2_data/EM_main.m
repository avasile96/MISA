clear all;
clc;
img = load_untouch_nii('T1.nii');
[mask,mu,v,p]=EMSeg(img.img(:,:,24),5);
