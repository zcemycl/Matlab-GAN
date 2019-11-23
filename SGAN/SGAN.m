clear all; close all; clc;
%% Semi-Supervised Generative Adversarial Network
%% Load Data
load('mnistAll.mat')
trainX = preprocess(mnist.train_images); 
trainY = mnist.train_labels;
testX = preprocess(mnist.test_images); 
testY = mnist.test_labels;
%% Settings
settings.latentDim = 100; settings.num_classes = 10;
settings.batch_size = 32; settings.image_size = [28,28,1]; 
settings.lr = 0.0002; settings.beta1 = 0.5;
settings.loss_weights = [.5,.5];
settings.beta2 = 0.999; settings.maxepochs = 50;

%% Initialization
%% Generator
paramsGen.FCW1 = dlarray(initializeGaussian([128*7*7,...
    settings.latentDim]));
paramsGen.FCb1 = dlarray(zeros(128*7*7,1,'single'));
paramsGen.BNo1 = dlarray(zeros(128,1,'single'));
paramsGen.BNs1 = dlarray(ones(128,1,'single'));
paramsGen.TCW1 = dlarray(initializeGaussian([3,3,128,128]));
paramsGen.TCb1 = dlarray(zeros(128,1,'single'));
paramsGen.BNo2 = dlarray(zeros(128,1,'single'));
paramsGen.BNs2 = dlarray(ones(128,1,'single'));
paramsGen.TCW2 = dlarray(initializeGaussian([3,3,64,128]));
paramsGen.TCb2 = dlarray(zeros(64,1,'single'));
paramsGen.BNo3 = dlarray(zeros(64,1,'single'));
paramsGen.BNs3 = dlarray(ones(64,1,'single'));
paramsGen.CNW1 = dlarray(initializeGaussian([3,3,64,1]));
paramsGen.CNb1 = dlarray(zeros(1,1,'single'));
stGen.BN1 = []; stGen.BN2 = []; stGen.BN3 = [];

%% Discriminator
paramsDis.CNW1 = dlarray(initializeGaussian([3,3,1,32]));
paramsDis.CNb1 = dlarray(zeros(32,1,'single'));
paramsDis.CNW2 = dlarray(initializeGaussian([3,3,32,64]));
paramsDis.CNb2 = dlarray(zeros(64,1,'single'));
paramsDis.BNo1 = dlarray(zeros(64,1,'single'));
paramsDis.BNs1 = dlarray(ones(64,1,'single'));
paramsDis.CNW3 = dlarray(initializeGaussian([3,3,64,128]));
paramsDis.CNb3 = dlarray(zeros(128,1,'single'));
paramsDis.BNo2 = dlarray(zeros(128,1,'single'));
paramsDis.BNs2 = dlarray(ones(128,1,'single'));
paramsDis.CNW4 = dlarray(initializeGaussian([3,3,128,256]));
paramsDis.CNb4 = dlarray(zeros(256,1,'single'));
paramsDis.FCW1 = dlarray(initializeGaussian([settings.num_classes+2,256*4*4]));
paramsDis.FCb1 = dlarray(zeros(settings.num_classes+2,1,'single'));
stDis.BN1 = []; stDis.BN2 = []; stDis.BN3 = [];

% average Gradient and average Gradient squared holders
avgG.Dis = []; avgGS.Dis = []; avgG.Gen = []; avgGS.Gen = [];
%% Train
numIterations = floor(size(trainX,4)/settings.batch_size);
out = false; epoch = 0; global_iter = 0;
while ~out
    tic; 
    shuffleid = randperm(size(trainX,4));
    trainXshuffle = trainX(:,:,:,shuffleid);
    trainYshuffle = trainY(shuffleid);
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        noise = gpdl(randn([settings.latentDim,...
            settings.batch_size]),'CB');
        idx = (i-1)*settings.batch_size+1:i*settings.batch_size;
        XBatch=gpdl(single(trainXshuffle(:,:,:,idx)),'SSCB');
        YBatch=gpdl(single(trainYshuffle(idx)),'B');

        [GradGen,GradDis,stGen,stDis] = ...
                dlfeval(@modelGradients,XBatch,YBatch,noise,...
                paramsGen,paramsDis,stGen,stDis);

        % Update Discriminator network parameters
        [paramsDis,avgG.Dis,avgGS.Dis] = ...
            adamupdate(paramsDis, GradDis, ...
            avgG.Dis, avgGS.Dis, global_iter, ...
            settings.lr, settings.beta1, settings.beta2);

        % Update Generator network parameters
        [paramsGen,avgG.Gen,avgGS.Gen] = ...
            adamupdate(paramsGen, GradGen, ...
            avgG.Gen, avgGS.Gen, global_iter, ...
            settings.lr, settings.beta1, settings.beta2);
        
        if i==1 || rem(i,20)==0
            progressplot(paramsGen,stGen,settings);
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
%% one hot encoding
function ohe = onehotencoding(labels,numLabels)
numBatch = length(labels);
ohe = zeros(numLabels,numBatch);

for i = 1:numBatch
    ohe(labels(i)+1,i)=1;
end
ohe(1:10,:) = 5/8*ohe(1:10,:);
ohe(end,:) = 1/16*ohe(end,:);

end
%% preprocess
function x = preprocess(x)
x = double(x)/255;
x = (x-.5)/.5;
x = reshape(x,28,28,1,[]);
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
%% Generator
function [dly,st] = Generator(dlx,params,st)
% fully connected
dly = fullyconnect(dlx,params.FCW1,params.FCb1);
dly = relu(dly);
% transposed convolution
dly = gpdl(reshape(dly,7,7,128,[]),'SSCB');
if isempty(st.BN1)
    [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,...
        params.BNo1,params.BNs1,'MeanDecay',0.8);
else
    [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,params.BNo1,...
        params.BNs1,st.BN1.mu,st.BN1.sig,...
        'MeanDecay',.8);
end
dly = dltranspconv(dly,params.TCW1,params.TCb1,...
    'Stride',2,'Cropping','same');
dly = relu(dly);
if isempty(st.BN2)
    [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,...
        params.BNo2,params.BNs2,'MeanDecay',0.8);
else
    [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,...
        params.BNs2,st.BN2.mu,st.BN2.sig,...
        'MeanDecay',.8);
end

dly = dltranspconv(dly,params.TCW2,params.TCb2,...
    'Stride',2,'Cropping','same');
dly = relu(dly);
if isempty(st.BN3)
    [dly,st.BN3.mu,st.BN3.sig] = batchnorm(dly,...
        params.BNo3,params.BNs3,'MeanDecay',0.8);
else
    [dly,st.BN3.mu,st.BN3.sig] = batchnorm(dly,params.BNo3,...
        params.BNs3,st.BN3.mu,st.BN3.sig,...
        'MeanDecay',.8);
end

dly = dlconv(dly,params.CNW1,params.CNb1,...
            'Padding','same');
% tanh
dly = tanh(dly);
end
%% Discriminator
function [dly,st] = Discriminator(dlx,params,st)
% convolution
%1
dly = dlconv(dlx,params.CNW1,params.CNb1,...
            'Stride',2,'Padding','same');
dly = leakyrelu(dly,0.2);
dly = dropout(dly,.25);
%2
dly = dlconv(dly,params.CNW2,params.CNb2,...
            'Stride',2,'Padding','same');
dly = leakyrelu(dly,0.2);
dly = dropout(dly,.25);
if isempty(st.BN1)
    [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,...
        params.BNo1,params.BNs1,'MeanDecay',0.8);
else
    [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,params.BNo1,...
        params.BNs1,st.BN1.mu,st.BN1.sig,...
        'MeanDecay',0.8);
end
%3
dly = dlconv(dly,params.CNW3,params.CNb3,...
            'Stride',2,'Padding','same');
dly = leakyrelu(dly,0.2);
dly = dropout(dly,.25);
if isempty(st.BN2)
    [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,...
        params.BNo2,params.BNs2,'MeanDecay',0.8);
else
    [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,...
        params.BNs2,st.BN2.mu,st.BN2.sig,...
        'MeanDecay',0.8);
end
%4
dly = dlconv(dly,params.CNW4,params.CNb4,...
            'Stride',1,'Padding','same');
dly = leakyrelu(dly,0.2);
dly = dropout(dly,.25);

% Fully connected
dly = gpdl(reshape(dly,4*4*256,[]),'CB');
dly = fullyconnect(dly,params.FCW1,params.FCb1);
% sigmoid
dly(1,:) = sigmoid(dly(1,:));
% softmax
dly(2:end,:) = softmax(dly(2:end,:));
end
%% modelGradients
function [GradGen,GradDis,stGen,stDis]=modelGradients(x,y,z,paramsGen,...
    paramsDis,stGen,stDis)
[fake_images,stGen] = Generator(z,paramsGen,stGen);
d_output_real = Discriminator(x,paramsDis,stDis);
[d_output_fake,stDis] = Discriminator(fake_images,paramsDis,stDis);
labels_real = onehotencoding(y,11);
labels_fake = onehotencoding(10*ones(length(y),1),11);

% Loss due to true or not
dlossreal=-mean(log(d_output_real(1,:)+eps)+...
    labels_real.*log(d_output_real(2:end,:)+eps),'all');
dlossfake=-mean(log(1-d_output_fake(1,:)+eps)+...
    labels_fake.*log(d_output_fake(2:end,:)+eps),'all');
d_loss = .5*(dlossreal+dlossfake);
% g_loss=-mean(log(d_output_fake(1,:)+eps)+...
%     labels_real.*log(d_output_fake(2:end,:)+eps),'all');
g_loss=-mean(log(d_output_fake(1,:)+eps),'all');

% For each network, calculate the gradients with respect to the loss.
GradGen = dlgradient(g_loss,paramsGen,'RetainData',true);
GradDis = dlgradient(d_loss,paramsDis);
end
%% progressplot
function progressplot(paramsGen,stGen,settings)
r = 5; c = 5;
noise = gpdl(randn([settings.latentDim,r*c]),'CB');
gen_imgs = Generator(noise,paramsGen,stGen);
gen_imgs = reshape(gen_imgs,28,28,[]);

fig = gcf;
if ~isempty(fig.Children)
    delete(fig.Children)
end

I = imtile(gatext(gen_imgs));
I = rescale(I);
imagesc(I)
title("Generated Images")
colormap gray

drawnow;
end
%% dropout
function dly = dropout(dlx,p)
if nargin < 2
    p = .3;
end
[n,d] = rat(p);
mask = randi([1,d],size(dlx));
mask(mask<=n)=0;
mask(mask>n)=1;
dly = dlx.*mask;

end