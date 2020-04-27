function main_training()

for i=1:5
diry=[pwd '\Dataset\' num2str(i)];
   disp(' features Extraction.....');
   feature1=training(diry);
   if i==1
       out1=feature1;
       group=ones(size(feature1,1),1)*i;
   else
       group1=ones(size(feature1,1),1)*i;
       group=[group;group1];
       out1=[out1;feature1];
   end
   
end 

inputs=out1;
targets=group;
net = newff(inputs',targets',5);
net1 = train(net,inputs',targets');
outputs = net(out1');
errors = outputs - group;
perf = 1-perform(net,outputs,group)
save('net_num1.mat','net1');
% Sim_train = sim(net1,inputs);
save Trained;
msgbox('Training Completed');

end


function out1=training(diry)

       file=dir(diry);
       out1=[];
for i1=3:length(file)
    
       filename=[diry '\' file(i1).name];
       
       Image1 = imread(filename);
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
    
    se = strel('disk',2);
    Out_Image = imdilate(image_out,se);
     Out_Image = bwareaopen(Out_Image,100);
     Out_Image = imfill(Out_Image,'holes');
    Out_Imgage = imclose(Out_Image,se);
      
     seg_img =Out_Imgage;
     
 %% feature extraction
 if ndims(Image2)==3
     gra = rgb2gray(Image2);
 else
     gra = Image2;
 end
glcms = graycomatrix(gra);
glcms1 = glcms/100;

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
Kurtosis = kurtosis(double(seg_img(:)));
Skewness = skewness(double(seg_img(:)));
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
feat_disease = [flbp,glcms1(:)',Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];

       out1(i1-2,:)= feat_disease';
end


end