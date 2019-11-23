clear all; close all; clc;
%% Pixels-to-Pixels
%% Load Data
load('AppleOrange.mat')
trainA = preprocess(Alist);
trainB = preprocess(Blist);
%% Settings
gf = 32; df = 64; settings.disc_patch = [8,8,1];
settings.lambda_cycle = 10;
settings.lambda_id = .1*settings.lambda_cycle;
settings.batch_size = 1; settings.image_size=[128,128,3]; 
settings.lr = 0.0002; settings.beta1 = 0.5;
settings.beta2 = 0.999; settings.maxepochs = 200;

%% 2 Generator (conv->deconv)
paramsGAB = InitializeGen(settings,gf);
paramsGBA = InitializeGen(settings,gf);
%% 2 Discriminator (conv)
paramsDA = InitializeDis(settings,df);
paramsDB = InitializeDis(settings,df);
%% Train
% average Gradient and average Gradient squared holders
avgG.DA = []; avgGS.DA = []; avgG.DB = []; avgGS.DB = []; 
avgG.GAB = []; avgGS.GAB = []; avgG.GBA = []; avgGS.GBA = [];
numIterations = floor(size(trainA,4)/settings.batch_size);
out = false; epoch = 0; global_iter = 0;
while ~out
    tic; 
    shuffleid = randperm(size(trainA,4));
    trainAshuffle = trainA(:,:,:,shuffleid);
    trainBshuffle = trainB(:,:,:,shuffleid);
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        ABatch=gpdl(trainAshuffle(:,:,:,i),'SSCB');
        BBatch=gpdl(trainBshuffle(:,:,:,i),'SSCB');

        [GradGAB,GradGBA,GradDA,GradDB] = ...
            dlfeval(@modelGradients,ABatch,BBatch,...
            paramsGAB,paramsGBA,paramsDA,paramsDB,...
            settings);

        % Update Discriminator network parameters
        [paramsDA,avgG.DA,avgGS.DA] = ...
            adamupdate(paramsDA, GradDA, ...
            avgG.DA, avgGS.DA, global_iter, ...
            settings.lr, settings.beta1, settings.beta2);
        [paramsDB,avgG.DB,avgGS.DB] = ...
            adamupdate(paramsDB, GradDB, ...
            avgG.DB, avgGS.DB, global_iter, ...
            settings.lr, settings.beta1, settings.beta2);

        % Update Generator network parameters
        [paramsGAB,avgG.GAB,avgGS.GAB] = ...
            adamupdate(paramsGAB, GradGAB, ...
            avgG.GAB, avgGS.GAB, global_iter, ...
            settings.lr, settings.beta1, settings.beta2);
        [paramsGBA,avgG.GBA,avgGS.GBA] = ...
            adamupdate(paramsGBA, GradGBA, ...
            avgG.GBA, avgGS.GBA, global_iter, ...
            settings.lr, settings.beta1, settings.beta2);
        if i==1 || rem(i,20)==0
            idxPlot = [200];
            APlot = gpdl(trainA(:,:,:,idxPlot),'SSCB');
            BPlot = gpdl(trainB(:,:,:,idxPlot),'SSCB');
            progressplot(APlot,BPlot,paramsGAB,paramsGBA)
        end
    
    end

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s")
    epoch = epoch+1;
    if epoch == settings.maxepochs
        out = true;
    end    
end

%% Helper Functions
%% progressplot
function progressplot(A,B,paramsGAB,paramsGBA)
fakeA = Generator(B,paramsGBA);
fakeB = Generator(A,paramsGAB);
reconB = Generator(fakeA,paramsGAB);
reconA = Generator(fakeB,paramsGBA);

fig = gcf;
if ~isempty(fig.Children)
    delete(fig.Children)
end

All = cat(4,A,fakeB,reconA,B,fakeA,reconB);
I = imtile(gatext(All),'GridSize',[2,3]);
I = rescale(I);
imagesc(I)
set(gca,'visible','off')

drawnow;
end
%% modelGradients
function [GradGAB,GradGBA,GradDA,GradDB]=...
    modelGradients(A,B,paramsGAB,paramsGBA,...
    paramsDA,paramsDB,settings)
fakeA = Generator(B,paramsGBA);
fakeB = Generator(A,paramsGAB);
reconB = Generator(fakeA,paramsGAB);
reconA = Generator(fakeB,paramsGBA);
idA = Generator(A,paramsGBA);
idB = Generator(B,paramsGAB);
validA0= Discriminator(A,paramsDA);
validB0= Discriminator(B,paramsDB);
validA = Discriminator(fakeA,paramsDA);
validB = Discriminator(fakeB,paramsDB);

% Loss calculation for Discriminator
%A
dA_loss_real = mean((validA0-1).^2,'all');
dA_loss_fake = mean((validA).^2,'all');
dA_loss = .5*(dA_loss_real+dA_loss_fake);
%B
dB_loss_real = mean((validB0-1).^2,'all');
dB_loss_fake = mean((validB).^2,'all');
dB_loss = .5*(dB_loss_real+dB_loss_fake);

% Loss calculation for Generator
%AB
gAB_loss_fake = mean((validB-1).^2,'all');
gAB_L1id = mean(abs(idB-B),'all');
gAB_L1re = mean(abs(reconB-B),'all');
%BA
gBA_loss_fake = mean((validA-1).^2,'all');
gBA_L1id = mean(abs(idA-A),'all');
gBA_L1re = mean(abs(reconA-A),'all');
%Total
g_loss=gAB_loss_fake+gBA_loss_fake+...
    settings.lambda_cycle*(gAB_L1re+gBA_L1re)+...
    settings.lambda_id*(gAB_L1id+gBA_L1id);

[GradGAB,GradGBA] = dlgradient(g_loss,...
    paramsGAB,paramsGBA,...
    'RetainData',true);
GradDA = dlgradient(dA_loss,paramsDA);
GradDB = dlgradient(dB_loss,paramsDB);
end
%% Generator
function dly = Generator(dlx,params)
d0 = dlx;
evaconv='d%d = dlconv(d%d,params.CNW%d,params.CNb%d,"Stride",2,"Padding","same");';
evaleak='d%d = leakyrelu(d%d,.2);';
evainst='d%d = instancenorm(d%d);';
evatran='u%d=dltranspconv(u%d,params.TCW%d,params.TCb%d,"Stride",2,"Cropping","same");';
evacat3='u%d=cat(3,u%d,d%d);';
evainst2='u%d = instancenorm(u%d);';
for i = 1:4
    eval(sprintf(evaconv,i,i-1,i,i));
    eval(sprintf(evaleak,i,i));
    eval(sprintf(evainst,i,i));
end
u0 = d4;
% u1=dltranspconv(d0,params.TCW1,params.TCb1,'Stride',2,'Cropping','same');
for i = 0:3
    eval(sprintf(evatran,i+1,i,i+1,i+1));
    if i < 3
        eval(sprintf(evainst2,i+1,i+1));
        eval(sprintf(evacat3,i+1,i+1,3-i));
    end
end
% tanh layer
dly = tanh(u4);
end
%% Discriminator
function dly = Discriminator(dlx,params)
d0 = dlx;
evaconv='d%d = dlconv(d%d,params.CNW%d,params.CNb%d,"Stride",%d,"Padding","same");';
evaleak='d%d = leakyrelu(d%d,.2);';
evainst='d%d = instancenorm(d%d);';
for i = 1:5
    if i < 5
        eval(sprintf(evaconv,i,i-1,i,i,2))
    else
        eval(sprintf(evaconv,i,i-1,i,i,1))
    end
    eval(sprintf(evaleak,i,i))
    if i > 1
        eval(sprintf(evainst,i,i))
    end
end
dly = d5;
end
%% Instance normalization
function dly = instancenorm(dlx)
% x_ijkt (SSCB) Dimension [WHCT]
mukt = sum(dlx,[1,2])/prod(size(dlx,[1,2]));
m = [1:size(dlx,1)]';
varkt=sum((dlx-m.*mukt).^2,[1,2])/prod(size(dlx,[1,2]));
dly = (dlx-mukt)./sqrt(varkt+eps);
end
%% extract data
function x = gatext(x)
x = gather(extractdata(x));
end
%% gpu dl array wrapper
function dlx = gpdl(x,labels)
dlx = gpuArray(dlarray(x,labels));
end
%% Weight initialization
function parameter = initializeGaussian(parameterSize,sigma)
if nargin < 2
    sigma = 0.05;
end
parameter = randn(parameterSize, 'single') .* sigma;
end
%% preprocess
function x = preprocess(x)
% x = double(x)/255;
x = (x-.5)/.5;
x = reshape(x,128,128,3,[]);
end
%% params Generator Initilization
function paramsGen = InitializeGen(settings,gf)
paramsGen.CNW1 = dlarray(initializeGaussian([4,4,settings.image_size(3),gf]));
paramsGen.CNb1 = dlarray(zeros(gf,1,'single'));
paramsGen.CNW2 = dlarray(initializeGaussian([4,4,gf,2*gf]));
paramsGen.CNb2 = dlarray(zeros(2*gf,1,'single'));
paramsGen.CNW3 = dlarray(initializeGaussian([4,4,2*gf,4*gf]));
paramsGen.CNb3 = dlarray(zeros(4*gf,1,'single'));
paramsGen.CNW4 = dlarray(initializeGaussian([4,4,4*gf,8*gf]));
paramsGen.CNb4 = dlarray(zeros(8*gf,1,'single'));

paramsGen.TCW1 = dlarray(initializeGaussian([4,4,4*gf,8*gf]));
paramsGen.TCb1 = dlarray(zeros(4*gf,1,'single'));
paramsGen.TCW2 = dlarray(initializeGaussian([4,4,2*gf,8*gf]));
paramsGen.TCb2 = dlarray(zeros(2*gf,1,'single'));
paramsGen.TCW3 = dlarray(initializeGaussian([4,4,gf,4*gf]));
paramsGen.TCb3 = dlarray(zeros(gf,1,'single'));
paramsGen.TCW4 = dlarray(initializeGaussian([4,4,settings.image_size(3),2*gf]));
paramsGen.TCb4 = dlarray(zeros(settings.image_size(3),1,'single'));
end

function paramsDis= InitializeDis(settings,df)
paramsDis.CNW1 = dlarray(initializeGaussian([4,4,settings.image_size(3),df]));
paramsDis.CNb1 = dlarray(zeros(df,1,'single'));
paramsDis.CNW2 = dlarray(initializeGaussian([4,4,df,2*df]));
paramsDis.CNb2 = dlarray(zeros(2*df,1,'single'));
paramsDis.CNW3 = dlarray(initializeGaussian([4,4,2*df,4*df]));
paramsDis.CNb3 = dlarray(zeros(4*df,1,'single'));
paramsDis.CNW4 = dlarray(initializeGaussian([4,4,4*df,8*df]));
paramsDis.CNb4 = dlarray(zeros(8*df,1,'single'));
paramsDis.CNW5 = dlarray(initializeGaussian([4,4,8*df,1]));
paramsDis.CNb5 = dlarray(zeros(1,1,'single'));
end