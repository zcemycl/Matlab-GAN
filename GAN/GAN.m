clear all; close all; clc;
%% Basic Generative Adversarial Network
%% Load Data
load('mnistAll.mat')
trainX = preprocess(mnist.train_images); 
trainY = mnist.train_labels;
testX = preprocess(mnist.test_images); 
testY = mnist.test_labels;
%% Settings
settings.latent_dim = 100;
settings.batch_size = 32; settings.image_size = [28,28,1]; 
settings.lrD = 0.0002; settings.lrG = 0.0002; settings.beta1 = 0.5;
settings.beta2 = 0.999; settings.maxepochs = 50;

%% Initialization
%% Generator
paramsGen.FCW1 = dlarray(...
    initializeGaussian([256,settings.latent_dim],.02));
paramsGen.FCb1 = dlarray(zeros(256,1,'single'));
paramsGen.BNo1 = dlarray(zeros(256,1,'single'));
paramsGen.BNs1 = dlarray(ones(256,1,'single'));
paramsGen.FCW2 = dlarray(initializeGaussian([512,256]));
paramsGen.FCb2 = dlarray(zeros(512,1,'single'));
paramsGen.BNo2 = dlarray(zeros(512,1,'single'));
paramsGen.BNs2 = dlarray(ones(512,1,'single'));
paramsGen.FCW3 = dlarray(initializeGaussian([1024,512]));
paramsGen.FCb3 = dlarray(zeros(1024,1,'single'));
paramsGen.BNo3 = dlarray(zeros(1024,1,'single'));
paramsGen.BNs3 = dlarray(ones(1024,1,'single'));
paramsGen.FCW4 = dlarray(initializeGaussian(...
    [prod(settings.image_size),1024]));
paramsGen.FCb4 = dlarray(zeros(prod(settings.image_size)...
    ,1,'single'));

stGen.BN1 = []; stGen.BN2 = []; stGen.BN3 = [];

%% Discriminator
paramsDis.FCW1 = dlarray(initializeGaussian([1024,...
     prod(settings.image_size)],.02));
paramsDis.FCb1 = dlarray(zeros(1024,1,'single'));
paramsDis.BNo1 = dlarray(zeros(1024,1,'single'));
paramsDis.BNs1 = dlarray(ones(1024,1,'single'));
paramsDis.FCW2 = dlarray(initializeGaussian([512,1024]));
paramsDis.FCb2 = dlarray(zeros(512,1,'single'));
paramsDis.BNo2 = dlarray(zeros(512,1,'single'));
paramsDis.BNs2 = dlarray(ones(512,1,'single'));
paramsDis.FCW3 = dlarray(initializeGaussian([256,512]));
paramsDis.FCb3 = dlarray(zeros(256,1,'single'));
paramsDis.FCW4 = dlarray(initializeGaussian([1,256]));
paramsDis.FCb4 = dlarray(zeros(1,1,'single'));

stDis.BN1 = []; stDis.BN2 = [];

% average Gradient and average Gradient squared holders
avgG.Dis = []; avgGS.Dis = []; avgG.Gen = []; avgGS.Gen = [];
%% Train
numIterations = floor(size(trainX,2)/settings.batch_size);
out = false; epoch = 0; global_iter = 0;
while ~out
    tic; 
    trainXshuffle = trainX(:,randperm(size(trainX,2)));
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        noise = gpdl(randn([settings.latent_dim,...
            settings.batch_size]),'CB');
        idx = (i-1)*settings.batch_size+1:i*settings.batch_size;
        XBatch=gpdl(single(trainXshuffle(:,idx)),'CB');

        [GradGen,GradDis,stGen,stDis] = ...
                dlfeval(@modelGradients,XBatch,noise,...
                paramsGen,paramsDis,stGen,stDis);

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
%                   imwrite(imind,cm,'GANmnist.gif','gif', 'Loopcount',inf); 
%                 else 
%                   imwrite(imind,cm,'GANmnist.gif','gif','WriteMode','append'); 
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
x = reshape(x,28*28,[]);
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
%1
dly = fullyconnect(dlx,params.FCW1,params.FCb1);
dly = leakyrelu(dly,0.2);
% if isempty(st.BN1)
%     [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,params.BNo1,params.BNs1);
% else
%     [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,params.BNo1,...
%         params.BNs1,st.BN1.mu,st.BN1.sig);
% end
%2
dly = fullyconnect(dly,params.FCW2,params.FCb2);
dly = leakyrelu(dly,0.2);
% if isempty(st.BN2)
%     [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,params.BNs2);
% else
%     [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,...
%         params.BNs2,st.BN2.mu,st.BN2.sig);
% end
%3
dly = fullyconnect(dly,params.FCW3,params.FCb3);
dly = leakyrelu(dly,0.2);
% if isempty(st.BN3)
%     [dly,st.BN3.mu,st.BN3.sig] = batchnorm(dly,params.BNo3,params.BNs3);
% else
%     [dly,st.BN3.mu,st.BN3.sig] = batchnorm(dly,params.BNo3,...
%         params.BNs3,st.BN3.mu,st.BN3.sig);
% end
%4
dly = fullyconnect(dly,params.FCW4,params.FCb4);
% tanh
dly = tanh(dly);
end
%% Discriminator
function [dly,st] = Discriminator(dlx,params,st)
% fully connected 
%1
dly = fullyconnect(dlx,params.FCW1,params.FCb1);
dly = leakyrelu(dly,0.2);
dly = dropout(dly);
% if isempty(st.BN1)
%     [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,params.BNo1,params.BNs1);
% else
%     [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,params.BNo1,...
%         params.BNs1,st.BN1.mu,st.BN1.sig);
% end
%2
dly = fullyconnect(dly,params.FCW2,params.FCb2);
dly = leakyrelu(dly,0.2);
dly = dropout(dly);
% if isempty(st.BN2)
%     [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,params.BNs2);
% else
%     [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,...
%         params.BNs2,st.BN2.mu,st.BN2.sig);
% end
%3
dly = fullyconnect(dly,params.FCW3,params.FCb3);
dly = leakyrelu(dly,0.2);
dly = dropout(dly);
%4
dly = fullyconnect(dly,params.FCW4,params.FCb4);
% sigmoid
dly = sigmoid(dly);
end
%% modelGradients
function [GradGen,GradDis,stGen,stDis]=modelGradients(x,z,paramsGen,...
    paramsDis,stGen,stDis)
[fake_images,stGen] = Generator(z,paramsGen,stGen);
d_output_real = Discriminator(x,paramsDis,stDis);
[d_output_fake,stDis] = Discriminator(fake_images,paramsDis,stDis);

% Loss due to true or not
d_loss = -mean(.9*log(d_output_real+eps)+log(1-d_output_fake+eps));
g_loss = -mean(log(d_output_fake+eps));

% For each network, calculate the gradients with respect to the loss.
GradGen = dlgradient(g_loss,paramsGen,'RetainData',true);
GradDis = dlgradient(d_loss,paramsDis);
end
%% progressplot
function progressplot(paramsGen,stGen,settings)
r = 5; c = 5;
noise = gpdl(randn([settings.latent_dim,r*c]),'CB');
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
n = p*10;
mask = randi([1,10],size(dlx));
mask(mask<=n)=0;
mask(mask>n)=1;
dly = dlx.*mask;

end