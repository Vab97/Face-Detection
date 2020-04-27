clc;
clear;
close all;

%% Image Aquisition
[file,path] = uigetfile('*.*');
Image1 = imread([path,file]);



%% Resize Image
Image1 = imresize(Image1,[200 200]);
Image2 = imsharpen(Image1);
[M,N,k] = size(Image2);

 % Convert the image from RGB to YCbCr
    img_ycbcr = rgb2ycbcr(Image2);
    Cb = img_ycbcr(:,:,2);
    Cr = img_ycbcr(:,:,3);


% Expected the hand to be in the central of the image
    central_color = img_ycbcr(int32(M/2),int32(N/2),:);
    Cb_Color = central_color(:,:,2);
    Cr_Color = central_color(:,:,3);
    % Set the range
    Cb_Difference = 15;
    Cr_Difference = 10;
 
    % Detect skin pixels
    [r,c,v] = find(Cb>=Cb_Color-Cr_Difference & Cb<=Cb_Color+Cb_Difference & Cr>=Cr_Color-Cr_Difference & Cr<=Cr_Color+Cr_Difference);
    
    len = length(r);
    % Mark detected pixels
    for i=1:len
        
        if i<=len
        image_out(r(i),c(i)) = 1;
        
        else
         image_out(r(i),c(i)) = 0;
        end 
    end
    
    %% morphological operation
    se = strel('disk',2);
    Out_Image = imdilate(image_out,se);
    Out_Image = bwareaopen(Out_Image,100);
    Out_Image = imfill(Out_Image,'holes');
    Out_Imgage = imclose(Out_Image,se);
   
    
    
    figure(1)
    subplot(221),imshow(Image1),title('Input Image');
    subplot(222),imshow(Image2),title('Sharpen Image');
    subplot(223),imshow(img_ycbcr),title('YCbCr Image');
    %subplot(224),imshow(Out_Imgage),title('Black and white image');
  
    
  pause(0.01)

    
     %% feature extraction
     seg_img =Out_Imgage;
 if ndims(Image2)==3
     gra = rgb2gray(Image2);
 else
     gra = Image2;
 end
 %% GLCM feature extraction
 
glcms = graycomatrix(gra);
glcms1 = glcms/100;

%% LBP Feature extraction
flbp = extractLBPFeatures(gra);
% Derive Statistics from GLCM
stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(seg_img);
Standard_Deviation = std2(seg_img);
Entropy = entropy(seg_img);
RMS = mean2(rms(seg_img));
%Skewness = skewness(img)
Variance = mean2(var(double(seg_img)));
a = sum(double(seg_img(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis((seg_img(:)));
Skewness = skewness((seg_img(:)));
% Inverse Difference Movement
m = size(seg_img,1);
n = size(seg_img,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = seg_img(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff); 
feat_disease = [LBP(i,j),glcms1(:)',Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy,Variance, Smoothness, IDM];
%% classification using BPNN

load 'net_num1.mat'

prediction = round(net1(feat_disease));

if prediction==Deepika
    disp('Image indication Deepika Padukone');
    msgbox('Ambitious,Perfectionist,Discipliner,Traditional');
elseif prediction==Ranveer
    disp('Image indication Ranveer Singh');
    msgbox('Quirky fashion sense,Entrepreneurial,Courageous,Creative,Energtic');
elseif prediction==Vicky
    disp('Image indication Vicky Kaushal');
    msgbox('Ambitious, Enthuastic, Creative,Silent person ');
else
    disp('No Celeb Face Detected try again');
end



