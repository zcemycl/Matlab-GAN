clear all; close all; clc;
%% Auxiliary Classifier Generative Adversarial Network
%% Load Data
load('mnistAll.mat')
trainX = preprocess(mnist.train_images); 
trainY = mnist.train_labels;
testX = preprocess(mnist.test_images); 
testY = mnist.test_labels;
%% Settings
settings.latentDim = 100; settings.num_labels = 10;
settings.batch_size = 32; settings.image_size = [28,28,1]; 
settings.lrD = 0.0002; settings.lrG = 0.0002; settings.beta1 = 0.5;
settings.beta2 = 0.999; settings.maxepochs = 50;

%% Initialization
%% Generator
paramsGen.EMW1 = dlarray(...
    initializeGaussian([settings.latentDim,...
    settings.num_labels]));
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
paramsDis.CNW1 = dlarray(initializeGaussian([3,3,1,16]));
paramsDis.CNb1 = dlarray(zeros(16,1,'single'));
paramsDis.CNW2 = dlarray(initializeGaussian([3,3,16,32]));
paramsDis.CNb2 = dlarray(zeros(32,1,'single'));
paramsDis.BNo1 = dlarray(zeros(32,1,'single'));
paramsDis.BNs1 = dlarray(ones(32,1,'single'));

paramsDis.CNW3 = dlarray(initializeGaussian([3,3,32,64]));
paramsDis.CNb3 = dlarray(zeros(64,1,'single'));
paramsDis.BNo2 = dlarray(zeros(64,1,'single'));
paramsDis.BNs2 = dlarray(ones(64,1,'single'));

paramsDis.CNW4 = dlarray(initializeGaussian([3,3,64,128]));
paramsDis.CNb4 = dlarray(zeros(128,1,'single'));

paramsDis.FCW1 = dlarray(initializeGaussian([1+settings.num_labels,128*4*4]));
paramsDis.FCb1 = dlarray(zeros(1+settings.num_labels,1,'single'));
stDis.BN1 = []; stDis.BN2 = [];

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
                paramsGen,paramsDis,stGen,stDis,settings);

        % Update Discriminator network parameters
        [paramsDis,avgG.Dis,avgGS.Dis] = ...
            adamupdate(paramsDis, GradDis, ...
            avgG.Dis, avgGS.Dis, global_iter, ...
            settings.lrD, settings.beta1, settings.beta2);

        % Update Generator network parameters
        [paramsGen,avgG.Gen,avgGS.Gen] = ...
            adamupdate(paramsGen, GradGen, ...
            avgG.Gen, avgGS.Gen, global_iter, ...
            settings.lrG, settings.beta1, settings.beta2);
        
        if i==1 || rem(i,20)==0
            progressplot(paramsGen,stGen,settings);
%             if i==1 || (epoch>=0 && i==1) 
%                 h = gcf;
%                 % Capture the plot as an image 
%                 frame = getframe(h); 
%                 im = frame2im(frame); 
%                 [imind,cm] = rgb2ind(im,256); 
%                 % Write to the GIF File 
%                 if epoch == 0
%                   imwrite(imind,cm,'DCGANmnist.gif','gif', 'Loopcount',inf); 
%                 else 
%                   imwrite(imind,cm,'DCGANmnist.gif','gif','WriteMode','append'); 
%                 end 
%             end
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
function [dly,st] = Generator(dlx,labels,params,st)
dly = embedding(dlx,labels,params);

% fully connected
dly = fullyconnect(dly,params.FCW1,params.FCb1);
dly = relu(dly);
dly = gpdl(reshape(dly,7,7,128,[]),'SSCB');
if isempty(st.BN1)
    [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,...
        params.BNo1,params.BNs1,'MeanDecay',0.8);
else
    [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,params.BNo1,...
        params.BNs1,st.BN1.mu,st.BN1.sig,...
        'MeanDecay',.8);
end
% transposed convolution
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
dly = gpdl(reshape(dly,4*4*128,[]),'CB');
dly = fullyconnect(dly,params.FCW1,params.FCb1);
% sigmoid
dly(1,:) = sigmoid(dly(1,:));
% softmax
dly(2:end,:) = softmax(dly(2:end,:));
end
%% modelGradients
function [GradGen,GradDis,stGen,stDis]=modelGradients(x,y,z,paramsGen,...
    paramsDis,stGen,stDis,settings)
y0 = randi([0,9],[settings.batch_size,1]);
ohey0 = onehotencoding(y0,settings.num_labels);
ohey = onehotencoding(y,settings.num_labels);

[fake_images,stGen] = Generator(z,y0,paramsGen,stGen);
d_output_real = Discriminator(x,paramsDis,stDis);
[d_output_fake,stDis] = Discriminator(fake_images,paramsDis,stDis);

% Loss due to true or not
% d_loss = -mean(.9*log(d_output_real+eps)+log(1-d_output_fake+eps));
d_loss_real = -.5*mean(log(d_output_real(1,:)+eps))-...
    .5*mean(log(sum(d_output_real(2:end,:).*ohey,1))+eps);
d_loss_fake = -.5*mean(log(1-d_output_fake(1,:)+eps))-...
    .5*mean(log(sum(d_output_fake(2:end,:).*ohey0,1))+eps);
d_loss = (d_loss_real+d_loss_fake)/2;
g_loss = -.5*mean(log(d_output_fake(1,:)+eps))-...
    .5*mean(log(sum(d_output_fake(2:end,:).*ohey0,1))+eps);

% For each network, calculate the gradients with respect to the loss.
GradGen = dlgradient(g_loss,paramsGen,'RetainData',true);
GradDis = dlgradient(d_loss,paramsDis);
end
%% progressplot
function progressplot(paramsGen,stGen,settings)
r = 2; c = 5;
labels = gpdl(single([0:9]'),'B');
noise = gpdl(randn([settings.latentDim,r*c]),'CB');
gen_imgs = Generator(noise,labels,paramsGen,stGen);
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
%% embedding
function dly = embedding(dlx,labels,params)
% params EM W (latentDim,num_labels)
%               / (img_elements,num_labels)
maskW = params.EMW1(:,labels+1);
dly = dlx.*maskW;
end
%% one hot encoding
function ohe = onehotencoding(labels,numLabels)
numBatch = length(labels);
ohe = zeros(numLabels,numBatch);

for i = 1:numBatch
    ohe(labels(i)+1,i)=1;
end

end