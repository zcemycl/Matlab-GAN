clear all; close all; clc;
%% Load images from files
path = 'D:\44754\Documents\Data\CMP_facade_DB_base\base\';
filesjpg = dir([path '*.jpg']);
filespng = dir([path '*.png']);
jpglist = []; pnglist = [];
for i = 1:378
    imj=imresize(imread([path filesjpg(i).name]),[256,256]);
    imp=imresize(imread([path filespng(i).name]),[256,256]);
    jpglist = cat(4,jpglist,im2double(imj));
    pnglist = cat(4,pnglist,im2double(imp));
end

path = 'D:\44754\Documents\Data\CMP_facade_DB_extended\extended\';
filesjpg = dir([path '*.jpg']);
filespng = dir([path '*.png']);
for i = 1:228
    imj=imresize(imread([path filesjpg(i).name]),[256,256]);
    imp=imresize(imread([path filespng(i).name]),[256,256]);
    jpglist = cat(4,jpglist,im2double(imj));
    pnglist = cat(4,pnglist,im2double(imp));
end

save('ImgLib.mat','jpglist')
save('LatCode.mat','pnglist')