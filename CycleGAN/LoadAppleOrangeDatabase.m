clear all; close all; clc;
%% Load images from files
path = 'D:\44754\Documents\Data\apple2orange\trainA\';
filesA = dir([path '*.jpg']);
Alist = []; 
for i = 1:length(filesA)
    imj=imresize(imread([path filesA(i).name]),[128,128]);
    Alist = cat(4,Alist,im2double(imj));
end

path = 'D:\44754\Documents\Data\apple2orange\trainB\';
filesB = dir([path '*.jpg']);
Blist = [];
for i = 1:length(filesB)
    imj=imresize(imread([path filesB(i).name]),[128,128]);
    Blist = cat(4,Blist,im2double(imj));
end