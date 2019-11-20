clear all; close all; clc;
%% Basic Generative Adversarial Network
%% Load Data
load('mnistAll.mat')
trainX = preprocess(mnist.train_images); 
trainY = mnist.train_labels;
testX = preprocess(mnist.test_images); 
testY = mnist.test_labels;
%% Settings
args.maxepochs = 50; args.c_weight = 0.5; args.z_dim = 62; 
args.batch_size = 16; args.image_size = [28,28,1]; 
args.lrD = 0.0002; args.lrG = 0.001; args.beta1 = 0.5;
args.beta2 = 0.999; args.cc_dim = 1; args.dc_dim = 10; 
args.sample_size = 100;
%% Weights, Biases, Offsets and Scales
% Generator
paramsGen.FCW1 = dlarray(...
    initializeGaussian([1024,args.z_dim+args.cc_dim+args.dc_dim]));
paramsGen.FCb1 = dlarray(zeros(1024,1,'single'));
paramsGen.BNo1 = dlarray(zeros(1024,1,'single'));
paramsGen.BNs1 = dlarray(ones(1024,1,'single'));

paramsGen.FCW2 = dlarray(initializeGaussian([128*7*7,1024]));
paramsGen.FCb2 = dlarray(zeros(128*7*7,1,'single'));
paramsGen.BNo2 = dlarray(zeros(128*7*7,1,'single'));
paramsGen.BNs2 = dlarray(ones(128*7*7,1,'single'));

paramsGen.TCW1 = dlarray(initializeGaussian([4,4,64,128]));
paramsGen.TCb1 = dlarray(zeros(64,1,'single'));
paramsGen.BNo3 = dlarray(zeros(64,1,'single'));
paramsGen.BNs3 = dlarray(ones(64,1,'single'));
paramsGen.TCW2 = dlarray(initializeGaussian([4,4,1,64]));
paramsGen.TCb2 = dlarray(zeros(1,1,'single'));

% Discriminator
paramsDis.CNW1 = dlarray(initializeGaussian([4,4,1,64]));
paramsDis.CNb1 = dlarray(zeros(64,1,'single'));
paramsDis.CNW2 = dlarray(initializeGaussian([4,4,64,128]));
paramsDis.CNb2 = dlarray(zeros(128,1,'single'));
paramsDis.BNo1 = dlarray(zeros(128,1,'single'));
paramsDis.BNs1 = dlarray(ones(128,1,'single'));
paramsDis.FCW1 = dlarray(initializeGaussian([128,128*7*7]));
paramsDis.FCb1 = dlarray(zeros(128,1,'single'));
paramsDis.BNo2 = dlarray(zeros(128,1,'single'));
paramsDis.BNs2 = dlarray(ones(128,1,'single'));
paramsDis.FCW2 = dlarray(initializeGaussian([...
    1+args.cc_dim+args.dc_dim,128]));
paramsDis.FCb2 = dlarray(zeros(1+args.cc_dim+args.dc_dim,...
    1,'single'));

% States for Batch Norm
stDis.BN1 = []; stDis.BN2 = [];
stGen.BN1 = []; stGen.BN2 = []; stGen.BN3 = [];
% average Gradient and average Gradient squared holders
avgG.Dis = []; avgGS.Dis = []; avgG.Gen = []; avgGS.Gen = [];
    
%% Train
out = false; epoch = 0; global_iter = 0;
while ~out
    tic; epoch = epoch+1;
    XTrainshuffle = trainX(:,:,:,randperm(size(trainX,4)));
    fprintf('Epoch %d\n',epoch) 
    for i=1:size(XTrainshuffle,4)/args.batch_size
        global_iter = global_iter+1;
        idx = (i-1)*args.batch_size+1:i*args.batch_size;
        XBatch=gpdl(double(XTrainshuffle(:,:,:,idx)));

        [GradDis,GradGen,stDis,stGen] = ...
                dlfeval(@modelGradients,XBatch,genNoiseCDC(args),...
                paramsDis,paramsGen,args,stDis,stGen);

        % Update Discriminator network parameters
        [paramsDis,avgG.Dis,avgGS.Dis] = ...
            adamupdate(paramsDis, GradDis, ...
            avgG.Dis, avgGS.Dis, global_iter, ...
            args.lrD, args.beta1, args.beta2);

        % Update Generator network parameters
        [paramsGen,avgG.Gen,avgGS.Gen] = ...
            adamupdate(paramsGen, GradGen, ...
            avgG.Gen, avgGS.Gen, global_iter, ...
            args.lrG, args.beta1, args.beta2);
        
        if i==1 || rem(i,1000) == 0
            h = gcf;
            % Capture the plot as an image 
            frame = getframe(h); 
            im = frame2im(frame); 
            [imind,cm] = rgb2ind(im,256); 
            % Write to the GIF File 
            if epoch == 1 && i==1
              imwrite(imind,cm,'InfoGANmnist.gif','gif', 'Loopcount',inf); 
            else 
              imwrite(imind,cm,'InfoGANmnist.gif','gif','WriteMode','append'); 
            end 
        end


        if rem(i,25) == 0
            % progress plot
            progressplot(args,paramsGen,stGen)

            if rem(i,750) == 0
                % report progress
                [d_loss,g_loss] = reportprogress(XBatch,genNoiseCDC(args),paramsDis,...
                            paramsGen,args,stDis,stGen);
                disp("Iter "+i+" Generator Loss: "+gatext(g_loss)+...
                    " Discriminator Loss: "+gatext(d_loss))
            end
        end

    end

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s")

    if epoch == args.maxepochs
        out = true;
    end    
end

%% Help Functions
%% preprocess
function x = preprocess(x)
x = double(x)./255;
x = (x-.5)./.5;
x = reshape(x,28,28,1,[]);
end
%% Weight Initialization
function parameter = initializeGaussian(parameterSize)
parameter = randn(parameterSize, 'single') .* 0.01;
end
%% extract data
function x = gatext(x)
x = gather(extractdata(x));
end
%% gpu dl array wrapper
function dlx = gpdl(x,labels)
if nargin < 2
    labels='SSCB';
end
dlx = gpuArray(dlarray(single(x),labels));
end
%% Generator
function [dly,st] = Generator(dlx,params,st)
% fully connected group
dly = fullyconnect(dlx,params.FCW1,params.FCb1);
if isempty(st.BN1)
    [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,params.BNo1,params.BNs1);
else
    [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,params.BNo1,...
        params.BNs1,st.BN1.mu,st.BN1.sig);
end
dly = relu(dly);

dly = fullyconnect(dly,params.FCW2,params.FCb2);
if isempty(st.BN2)
    [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,params.BNs2);
else
    [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,...
        params.BNs2,st.BN2.mu,st.BN2.sig);
end
dly = relu(dly);

% transpose convolution group
dly = gpdl(reshape(dly,7,7,128,[]));
dly = dltranspconv(dly,params.TCW1,params.TCb1,...
    'Stride',2,'Cropping','same');
if isempty(st.BN3)
    [dly,st.BN3.mu,st.BN3.sig] = batchnorm(dly,params.BNo3,params.BNs3);
else
    [dly,st.BN3.mu,st.BN3.sig] = batchnorm(dly,params.BNo3,...
        params.BNs3,st.BN3.mu,st.BN3.sig);
end
dly = relu(dly);

dly = dltranspconv(dly,params.TCW2,params.TCb2,...
    'Stride',2,'Cropping','same');
dly = tanh(dly);
end
%% Discriminator
function [dly,st] = Discriminator(dlx,params,args,st)
% Convolution group
dly = dlconv(dlx,params.CNW1,params.CNb1,...
    'Stride',2,'Padding','same');
dly = leakyrelu(dly,0.1);
dly = dlconv(dly,params.CNW2,params.CNb2,...
    'Stride',2,'Padding','same');
if isempty(st.BN1)
    [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,params.BNo1,params.BNs1);
else
    [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,params.BNo1,...
        params.BNs1,st.BN1.mu,st.BN1.sig);
end
dly = leakyrelu(dly,0.1);

% fully connected group
dly = gpuArray(dlarray(reshape(dly,128*7*7,[]),'CB'));
dly = fullyconnect(dly,params.FCW1,params.FCb1);
if isempty(st.BN2)
    [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,params.BNs2);
else
    [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,...
        params.BNs2,st.BN2.mu,st.BN2.sig);
end
dly = leakyrelu(dly,0.1);
dly = fullyconnect(dly,params.FCW2,params.FCb2);

dly(1,:) = sigmoid(dly(1,:));
dly(args.cc_dim+2:end,:) = softmax(dly(args.cc_dim+2:end,:));

end
%% dropout
function dly = dropout(dlx,p)
if nargin < 2
    p = .5;
end
[n,d] = rat(p);
mask = randi([1,d],size(dlx));
mask(mask<=n)=0;
mask(mask>n)=1;
dly = dlx.*mask;
end
%% General Noise
function out = genNoiseCDC(args)
noi = randn([args.z_dim,args.batch_size]);
out = [noi;gen_cc([args.cc_dim,args.batch_size]);...
    gen_dc([args.dc_dim,args.batch_size])];
out = gpuArray(dlarray(out,'CB'));
end
%% Generate Discrete Code
function out = gen_dc(shape)
out = zeros(shape);
random_cate = randi([1,shape(1)],shape(2),1);
for i = 1:length(random_cate)
    out(random_cate(i),i) = 1;
end
end
%% Generate Continuous Code
function out = gen_cc(shape)
out = randn(shape)*0.5;
end
%% Progress Plot
function progressplot(args,paramsGen,stGen)
fixednoise = zeros(args.z_dim,args.sample_size);
tmp = zeros(args.cc_dim,args.sample_size);
for i = 1:10
    tmp(1,(i-1)*10+1:i*10) = linspace(-2,2,10);
end
cc = tmp;

tmp = zeros(args.dc_dim,args.sample_size);
for i = 1:10
    tmp(i,(i-1)*10+1:i*10) = 1;
end
dc = tmp;

fake_data = gpuArray(dlarray(cat(1,fixednoise,cc,dc),'CB'));
fake_images = extractdata(Generator(fake_data,paramsGen,stGen));

fig = gcf;
if ~isempty(fig.Children)
    delete(fig.Children)
end

I = imtile(fake_images);
I = rescale(I);
imagesc(I)
title("Generated Images")

drawnow;

end
%% Report Progress
function [d_loss,g_loss] = reportprogress(x,z,paramsDis,...
    paramsGen,args,stDis,stGen)
fake_images = Generator(z,paramsGen,stGen);
d_output_real = Discriminator(x,paramsDis,args,stDis);
d_output_fake = Discriminator(fake_images,paramsDis,args,stDis);

% Loss due to true or not
d_loss_a = -mean(log(d_output_real(1,:))+log(1-d_output_fake(1,:)));
g_loss_a = -mean(log(d_output_fake(1,:)));

% cc loss
output_cc = d_output_fake(2,:);
d_loss_cc = mean((output_cc/0.5).^2);

% softmax classification loss
output_dc = d_output_fake(3:end,:);
d_loss_dc = -(mean(sum(z(args.z_dim+args.cc_dim+1:end,:).*output_dc,1))+...
    mean(sum(z(args.z_dim+args.cc_dim+1:end,:).*z(args.z_dim+args.cc_dim+1:end,:),1)));

% Discriminator Loss
d_loss = d_loss_a+args.c_weight*d_loss_cc+d_loss_dc;
% Generator Loss
g_loss = g_loss_a+args.c_weight*d_loss_cc+d_loss_dc;
end
%% Model Gradients
function [GradDis,GradGen,stDis,stGen] = modelGradients(x,z,paramsDis,...
    paramsGen,args,stDis,stGen)
[fake_images,stGen] = Generator(z,paramsGen,stGen);
d_output_real = Discriminator(x,paramsDis,args,stDis);
[d_output_fake,stDis] = Discriminator(fake_images,paramsDis,args,stDis);

% Loss due to true or not
d_loss_a = -mean(log(d_output_real(1,:))+log(1-d_output_fake(1,:)));
g_loss_a = -mean(log(d_output_fake(1,:)));

% cc loss
output_cc = d_output_fake(2,:);
d_loss_cc = mean((output_cc/0.5).^2);

% softmax classification loss
output_dc = d_output_fake(3:end,:);
d_loss_dc = -(mean(sum(z(args.z_dim+args.cc_dim+1:end,:).*output_dc,1))+...
    mean(sum(z(args.z_dim+args.cc_dim+1:end,:).*z(args.z_dim+args.cc_dim+1:end,:),1)));

% Discriminator Loss
d_loss = d_loss_a+args.c_weight*d_loss_cc+d_loss_dc;
% Generator Loss
g_loss = g_loss_a+args.c_weight*d_loss_cc+d_loss_dc;

% For each network, calculate the gradients with respect to the loss.
GradGen = dlgradient(g_loss,paramsGen,'RetainData',true);
GradDis = dlgradient(d_loss,paramsDis);

end