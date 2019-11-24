clear all; close all; clc;
%% Info Generative Adversarial Network
%% Settings
args.maxepochs = 5; args.c_weight = 1; args.z_dim = 128; 
args.batch_size = 16; args.image_size = [64,64,3]; 
args.lrD = 0.0002; args.lrG = 0.001; args.beta1 = 0.5;
args.beta2 = 0.999; args.cc_dim = 1; args.dc_dim = 10; 
args.sample_size = 100;
%% Weights, Biases, Offsets and Scales
%% Generator
[paramsGen,stGen] = initializeGen(args);
%% Discriminator
[paramsDis,stDis] = initializeDis(args);
%% Train
out = false; epoch = 0; global_iter = 0; count = 1;
% average Gradient and average Gradient squared holders
avgG.Dis = []; avgGS.Dis = []; avgG.Gen = []; avgGS.Gen = [];

while ~out
    tic; epoch = epoch+1;
    shuffleid = randperm(100000);
    fprintf('Epoch %d\n',epoch) 
    for i=1:floor(length(shuffleid)/args.batch_size)
        global_iter = global_iter+1;
        idx = (i-1)*args.batch_size+1:i*args.batch_size;
        XBatch=miniBatch(shuffleid(idx));

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

        if i==1 || rem(i,20) == 0
            % progress plot
            progressplot(args,paramsGen,stGen)
        end
        
        if rem(global_iter,count^2)==0 || rem(global_iter,500)==0
            h = gcf;
            % Capture the plot as an image 
            frame = getframe(h); 
            im = frame2im(frame); 
            [imind,cm] = rgb2ind(im,256); 
            % Write to the GIF File 
            if count == 1
              imwrite(imind,cm,'InfoGANcelebA.gif','gif', 'Loopcount',inf); 
            else 
              imwrite(imind,cm,'InfoGANcelebA.gif','gif','WriteMode','append'); 
            end 
            
            count = count+1;
        end

    end

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s")

    if epoch == args.maxepochs
        out = true;
    end    
end

%% Help Functions
%% miniBatch 
function dlx = miniBatch(ids)
path = 'D:\44754\Documents\Data\celeba-dataset\img_align_celeba\img_align_celeba\';
filesA = dir([path '*.jpg']);
dlx = [];
for i = 1:length(ids)
    imj=imresize(imread([path filesA(ids(i)).name]),[64,64]);
    imj=preprocess(imj);
    dlx = cat(4,dlx,im2double(imj));
end
dlx = gpdl(dlx);
end
%% preprocess
function x = preprocess(x)
x = double(x)./255;
x = (x-.5)./.5;
x = reshape(x,64,64,3,[]);
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
dly = dlx;
exstatbatch='[dly,st] = batchnormwrap(dly,params,st,%d);'; 
exstattran = 'dly = dltranspconv(dly,params.TCW%d,params.TCb%d,"Stride",%d,"Cropping",%d);';
exstatrelu = 'dly = relu(dly);';

for i = 1:5
    if i==1
        eval(sprintf(exstattran,i,i,1,0))
    elseif i==5
        eval(sprintf(exstattran,i,i,2,1))
        dly = tanh(dly);
    else
        eval(sprintf(exstattran,i,i,2,1))
        eval(sprintf(exstatbatch,i-1))
        eval(exstatrelu)
    end
end
end
%% Discriminator
function [dly,st] = Discriminator(dlx,params,args,st)
dly = dlx;
exstatconv ='dly = dlconv(dly,params.CNW%d,params.CNb%d,"Stride",%d,"Padding",%d);';
exstatleak = 'dly = leakyrelu(dly,.1);';
exstatbatch='[dly,st] = batchnormwrap(dly,params,st,%d);'; 

for i = 1:5
    if i == 1
        eval(sprintf(exstatconv,i,i,2,1))
        eval(exstatleak)
    elseif i == 5
        eval(sprintf(exstatconv,i,i,1,0))
    else
        eval(sprintf(exstatconv,i,i,2,1))
        eval(sprintf(exstatbatch,i-1))
        eval(exstatleak)      
    end
end
dly = gpdl(reshape(dly,1+args.cc_dim+args.dc_dim,[]),'CB');
dly(1,:) = sigmoid(dly(1,:));
dly(args.cc_dim+2:end,:) = softmax(dly(args.cc_dim+2:end,:));

end
%% General Noise
function out = genNoiseCDC(args)
noi = randn([args.z_dim,args.batch_size]);
out = [noi;gen_cc([args.cc_dim,args.batch_size]);...
    gen_dc([args.dc_dim,args.batch_size])];
out = reshape(out,1,1,args.z_dim+args.cc_dim+args.dc_dim,[]);
out = gpuArray(dlarray(out,'SSCB'));
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

fake_data = gpdl(reshape(cat(1,fixednoise,cc,dc),1,1,...
    args.z_dim+args.cc_dim+args.dc_dim,[]));
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
z = gpdl(reshape(z,args.z_dim+args.cc_dim+args.dc_dim,[]),'CB');
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
%% Initialize Generator params
function [paramsGen,stGen] = initializeGen(args)
paramsGen.TCW1 = dlarray(initializeGaussian([4,4,1024,...
    args.z_dim+args.cc_dim+args.dc_dim]));
paramsGen.TCb1 = dlarray(zeros(1024,1,'single'));
paramsGen.TCW2 = dlarray(initializeGaussian([4,4,512,1024]));
paramsGen.TCb2 = dlarray(zeros(512,1,'single'));
paramsGen.BNo1 = dlarray(zeros(512,1,'single'));
paramsGen.BNs1 = dlarray(ones(512,1,'single'));
paramsGen.TCW3 = dlarray(initializeGaussian([4,4,256,512]));
paramsGen.TCb3 = dlarray(zeros(256,1,'single'));
paramsGen.BNo2 = dlarray(zeros(256,1,'single'));
paramsGen.BNs2 = dlarray(ones(256,1,'single'));
paramsGen.TCW4 = dlarray(initializeGaussian([4,4,128,256]));
paramsGen.TCb4 = dlarray(zeros(128,1,'single'));
paramsGen.BNo3 = dlarray(zeros(128,1,'single'));
paramsGen.BNs3 = dlarray(ones(128,1,'single'));
paramsGen.TCW5 = dlarray(initializeGaussian([4,4,3,128]));
paramsGen.TCb5 = dlarray(zeros(3,1,'single'));

% States for Batch Norm
stGen.BN1 = []; stGen.BN2 = []; stGen.BN3 = []; 
end
%% Initialize Discriminator params
function [paramsDis,stDis] = initializeDis(args)
paramsDis.CNW1 = dlarray(initializeGaussian([4,4,3,128]));
paramsDis.CNb1 = dlarray(zeros(128,1,'single'));
paramsDis.CNW2 = dlarray(initializeGaussian([4,4,128,256]));
paramsDis.CNb2 = dlarray(zeros(256,1,'single'));
paramsDis.BNo1 = dlarray(zeros(256,1,'single'));
paramsDis.BNs1 = dlarray(ones(256,1,'single'));
paramsDis.CNW3 = dlarray(initializeGaussian([4,4,256,512]));
paramsDis.CNb3 = dlarray(zeros(512,1,'single'));
paramsDis.BNo2 = dlarray(zeros(512,1,'single'));
paramsDis.BNs2 = dlarray(ones(512,1,'single'));
paramsDis.CNW4 = dlarray(initializeGaussian([4,4,512,1024]));
paramsDis.CNb4 = dlarray(zeros(1024,1,'single'));
paramsDis.BNo3 = dlarray(zeros(1024,1,'single'));
paramsDis.BNs3 = dlarray(ones(1024,1,'single'));
paramsDis.CNW5 = dlarray(initializeGaussian([4,4,1024,...
    1+args.cc_dim+args.dc_dim]));
paramsDis.CNb5 = dlarray(zeros(1+args.cc_dim+args.dc_dim,1,'single'));

% States for Batch Norm
stDis.BN1 = []; stDis.BN2 = []; stDis.BN3 = [];
end
%% batchnormwrap
function [dly,st] = batchnormwrap(dlx,params,st,num)
exstat1=sprintf('if isempty(st.BN%d),',num);
exstat2=sprintf('[dly,st.BN%d.mu,st.BN%d.sig]=batchnorm(dlx,params.BNo%d,params.BNs%d,"MeanDecay",0.8);else,',num*ones(1,4));
exstat3=sprintf('[dly,st.BN%d.mu,st.BN%d.sig]=batchnorm(dlx,params.BNo%d,params.BNs%d,st.BN%d.mu,st.BN%d.sig,"MeanDecay",0.8);end',num*ones(1,6));
eval(strcat(exstat1,exstat2,exstat3));
end