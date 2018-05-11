% %读入AVIRIS遥感数据 145*145*220bands
 row     = 145; col     = 145; bandnum = 220;
im1     = multibandread('92AV3C.lan',[145,145,bandnum],'uint16',128,'BIL','ieee-le');
im1 (:,:,220)    = []; %剔除水蒸气波段
im1(:,:,150:163) = []; %剔除水蒸气波段
im1(:,:,104:108) = []; %剔除水蒸气波段
bandnum = 200;
im     = im1;
%% 读入AVIRIS GIS 数据
imGIS = multibandread('92AV3GT.GIS',[145,145,1],'uint8',128,'BSQ','ieee-le');
imshow(imGIS);
figure;imshow(label2rgb(imGIS));
img1=im(:,:,20)/max(max(max(im)));
img2=im(:,:,80)/max(max(max(im)));
img3=im(:,:,170)/max(max(max(im)));
figure;
imshow((img1+img2+img3)/3);
%% 根据GIS类别号选出对应于每类的数据
clsNum = 16; %16类数据 
clsAll = [];
posAll = [];

for i = 1 : clsNum                  %统计各类元素数目
    [x,y] = find (imGIS == i);  
    Class_num(i) = size(x,1);
end

choose_index  = 1:1:16;

for i = 1 : size(choose_index,2)  
    [x,y] = find (imGIS == choose_index(i));  
    Class_num1(i) = size(x,1);
    pos{i} = [x,y]';   %各类的位置信息
    for j=1:length(pos{i})
        cls{i}(:,j)  = im(x(j),y(j),:); %各类的光谱信息
    end
    clsAll = [clsAll cls{i}];  %所有类的光谱信息
    posAll = [posAll pos{i}];    %所有类的位置信息
end
   
data.fet = clsAll';   %data.fet为导入的所有的数据 
save datafet.mat;
label = [];
for i = 1 : size(choose_index,2)  
    label = [label, repmat(i,[1, Class_num1(i)])];
end

data.lab = label';  %data.lab为导入的所有的label 
save datalab.mat