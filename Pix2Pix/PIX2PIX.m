clear all; close all; clc;
%% Load Data
load('LatCode.mat')
load('ImgLib.mat')
trainX = preprocesscode(pnglist);
trainY = preprocess(jpglist);
%% Settings
gf = 64; df = 64;
settings.disc_patch = [16,16,1];
settings.batch_size = 1; settings.image_size = [256,256,3]; 
settings.lrD = 0.0002; settings.lrG = 0.0002; settings.beta1 = 0.5;
settings.beta2 = 0.999; settings.maxepochs = 200;

%% Generator (conv->deconv)
[paramsGen,stGen] = InitializeGen(settings,gf);
%% Discriminator (conv)
[paramsDis,stDis] = InitializeDis(settings,df);
%% Train
% average Gradient and average Gradient squared holders
avgG.Dis = []; avgGS.Dis = []; avgG.Gen = []; avgGS.Gen = [];
numIterations = floor(size(trainX,4)/settings.batch_size);
out = false; epoch = 0; global_iter = 0;
while ~out
    tic; 
    shuffleid = randperm(size(trainX,4));
    trainXshuffle = trainX(:,:,:,shuffleid);
    trainYshuffle = trainY(:,:,:,shuffleid);
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        XBatch=gpdl(trainXshuffle(:,:,:,i),'SSCB');
        YBatch=gpdl(trainYshuffle(:,:,:,i),'SSCB');

        [GradGen,GradDis,stGen,stDis] = ...
                dlfeval(@modelGradients,XBatch,YBatch,...
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
            idxPlot = [1,201,401];
            XPlot = gpdl(trainX(:,:,:,idxPlot),'SSCB');
            YPlot = gpdl(trainY(:,:,:,idxPlot),'SSCB');
            progressplot(XPlot,YPlot,paramsGen,stGen);
            
            if i==1 || rem(i,200)==0
                h = gcf;
                % Capture the plot as an image 
                frame = getframe(h); 
                im = frame2im(frame); 
                [imind,cm] = rgb2ind(im,256); 
                % Write to the GIF File 
                if epoch == 0
                  imwrite(imind,cm,'p2pfacade.gif','gif', 'Loopcount',inf); 
                else 
                  imwrite(imind,cm,'p2pfacade.gif','gif','WriteMode','append'); 
                end 
            end
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
function progressplot(x,y,paramsGen,stGen)
genImg = Generator(x,paramsGen,stGen);

fig = gcf;
if ~isempty(fig.Children)
    delete(fig.Children)
end

subplot(3,1,1)
I = imtile(gatext(x),'GridSize',[1,3]);
I = rescale(I);
imagesc(I)
set(gca,'visible','off')
title("Input")

subplot(3,1,2)
I = imtile(gatext(genImg),'GridSize',[1,3]);
I = rescale(I);
imagesc(I)
set(gca,'visible','off')
title("Output")

subplot(3,1,3)
I = imtile(gatext(y),'GridSize',[1,3]);
I = rescale(I);
imagesc(I)
set(gca,'visible','off')
title("Truth")

drawnow;
end
%% modelGradients
function [GradGen,GradDis,stGen,stDis]=modelGradients(x,y,paramsGen,paramsDis,stGen,stDis)
pairYes = Discriminator(cat(3,y,x),paramsDis,stDis);
[fakeA,stGen] = Generator(x,paramsGen,stGen);
[pairNo,stDis] = Discriminator(cat(3,fakeA,x),paramsDis,stDis); 

d_loss=-.5*(mean(log(pairYes+eps)+log(1-pairNo+eps),'all'));
g_loss=-.5*mean(log(pairNo+eps),'all')+100*mean(abs(y-fakeA),'all'); 

GradGen = dlgradient(g_loss,paramsGen,'RetainData',true);
GradDis = dlgradient(d_loss,paramsDis);
end
%% Generator
function [dly,st] = Generator(dlx,params,st)
dly = dlconv(dlx,params.CNW1,params.CNb1,...
            'Stride',2,'Padding','same');
d1 = leakyrelu(dly,.2);

exstatconv ='d%d = dlconv(d%d,params.CNW%d,params.CNb%d,"Stride",2,"Padding","same");';
exstatleak = 'd%d = leakyrelu(d%d,.2);';
exstatbatch='[d%d,st] = batchnormwrap(d%d,params,st,%d);'; 
exstattran = 'u%d = dltranspconv(u%d,params.TCW%d,params.TCb%d,"Stride",2,"Cropping","same");';
exstatrelu = 'u%d = relu(u%d);';
exstatbatch2='[u%d,st] = batchnormwrap(u%d,params,st,%d);'; 
exstatcat = 'u%d=cat(3,u%d,d%d);';
for i = 2:7
    eval(sprintf(exstatconv,i,i-1,i*ones(1,2)));
    eval(sprintf(exstatleak,i,i));
    eval(sprintf(exstatbatch,i,i,i-1));
end

u1=dltranspconv(d7,params.TCW1,params.TCb1,'Stride',2,'Cropping','same');
u1=batchnormwrap(u1,params,st,7);
u1=cat(3,u1,d6);
for i = 2:7
    eval(sprintf(exstattran,i,i-1,i,i));
    if i ~= 7 
        eval(sprintf(exstatrelu,i,i));
        eval(sprintf(exstatbatch2,i,i,6+i));
        eval(sprintf(exstatcat,i,i,7-i));
    end
end

dly = tanh(u7);
end
%% Discriminator
function [dly,st] = Discriminator(dlx,params,st)
dly = dlconv(dlx,params.CNW1,params.CNb1,...
            'Stride',2,'Padding','same');
dly = leakyrelu(dly,.2);
dly = dlconv(dly,params.CNW2,params.CNb2,...
            'Stride',2,'Padding','same');
dly = leakyrelu(dly,.2);
[dly,st] = batchnormwrap(dly,params,st,1);
dly = dlconv(dly,params.CNW3,params.CNb3,...
            'Stride',2,'Padding','same');
dly = leakyrelu(dly,.2);
[dly,st] = batchnormwrap(dly,params,st,2);
dly = dlconv(dly,params.CNW4,params.CNb4,...
            'Stride',2,'Padding','same');
dly = leakyrelu(dly,.2);
[dly,st] = batchnormwrap(dly,params,st,3);
dly = dlconv(dly,params.CNW5,params.CNb5,...
            'Stride',1,'Padding','same');
dly = sigmoid(dly);
end
%% Preprocess Code
function x = preprocesscode(x)
x = x/max(x(:));
end
%% batchnormwrap
function [dly,st] = batchnormwrap(dlx,params,st,num)
exstat1=sprintf('if isempty(st.BN%d),',num);
exstat2=sprintf('[dly,st.BN%d.mu,st.BN%d.sig]=batchnorm(dlx,params.BNo%d,params.BNs%d,"MeanDecay",0.8);else,',num*ones(1,4));
exstat3=sprintf('[dly,st.BN%d.mu,st.BN%d.sig]=batchnorm(dlx,params.BNo%d,params.BNs%d,st.BN%d.mu,st.BN%d.sig,"MeanDecay",0.8);end',num*ones(1,6));
eval(strcat(exstat1,exstat2,exstat3));
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
x = reshape(x,256,256,3,[]);
end
%% params Generator Initilization
function [paramsGen,stGen] = InitializeGen(settings,gf)
paramsGen.CNW1 = dlarray(initializeGaussian([4,4,1,gf]));
paramsGen.CNb1 = dlarray(zeros(gf,1,'single'));
paramsGen.CNW2 = dlarray(initializeGaussian([4,4,gf,2*gf]));
paramsGen.CNb2 = dlarray(zeros(2*gf,1,'single'));
paramsGen.BNo1 = dlarray(zeros(2*gf,1,'single'));
paramsGen.BNs1 = dlarray(ones(2*gf,1,'single'));
paramsGen.CNW3 = dlarray(initializeGaussian([4,4,2*gf,4*gf]));
paramsGen.CNb3 = dlarray(zeros(4*gf,1,'single'));
paramsGen.BNo2 = dlarray(zeros(4*gf,1,'single'));
paramsGen.BNs2 = dlarray(ones(4*gf,1,'single'));
paramsGen.CNW4 = dlarray(initializeGaussian([4,4,4*gf,8*gf]));
paramsGen.CNb4 = dlarray(zeros(8*gf,1,'single'));
paramsGen.BNo3 = dlarray(zeros(8*gf,1,'single'));
paramsGen.BNs3 = dlarray(ones(8*gf,1,'single'));
paramsGen.CNW5 = dlarray(initializeGaussian([4,4,8*gf,8*gf]));
paramsGen.CNb5 = dlarray(zeros(8*gf,1,'single'));
paramsGen.BNo4 = dlarray(zeros(8*gf,1,'single'));
paramsGen.BNs4 = dlarray(ones(8*gf,1,'single'));
paramsGen.CNW6 = dlarray(initializeGaussian([4,4,8*gf,8*gf]));
paramsGen.CNb6 = dlarray(zeros(8*gf,1,'single'));
paramsGen.BNo5 = dlarray(zeros(8*gf,1,'single'));
paramsGen.BNs5 = dlarray(ones(8*gf,1,'single'));
paramsGen.CNW7 = dlarray(initializeGaussian([4,4,8*gf,8*gf]));
paramsGen.CNb7 = dlarray(zeros(8*gf,1,'single'));
paramsGen.BNo6 = dlarray(zeros(8*gf,1,'single'));
paramsGen.BNs6 = dlarray(ones(8*gf,1,'single'));

paramsGen.TCW1 = dlarray(initializeGaussian([4,4,8*gf,8*gf]));
paramsGen.TCb1 = dlarray(zeros(8*gf,1,'single'));
paramsGen.BNo7 = dlarray(zeros(8*gf,1,'single'));
paramsGen.BNs7 = dlarray(ones(8*gf,1,'single'));
paramsGen.TCW2 = dlarray(initializeGaussian([4,4,8*gf,16*gf]));
paramsGen.TCb2 = dlarray(zeros(8*gf,1,'single'));
paramsGen.BNo8 = dlarray(zeros(8*gf,1,'single'));
paramsGen.BNs8 = dlarray(ones(8*gf,1,'single'));

paramsGen.TCW3 = dlarray(initializeGaussian([4,4,8*gf,16*gf]));
paramsGen.TCb3 = dlarray(zeros(8*gf,1,'single'));
paramsGen.BNo9 = dlarray(zeros(8*gf,1,'single'));
paramsGen.BNs9 = dlarray(ones(8*gf,1,'single'));

paramsGen.TCW4 = dlarray(initializeGaussian([4,4,4*gf,16*gf]));
paramsGen.TCb4 = dlarray(zeros(4*gf,1,'single'));
paramsGen.BNo10 = dlarray(zeros(4*gf,1,'single'));
paramsGen.BNs10 = dlarray(ones(4*gf,1,'single'));
paramsGen.TCW5 = dlarray(initializeGaussian([4,4,2*gf,8*gf]));
paramsGen.TCb5 = dlarray(zeros(2*gf,1,'single'));
paramsGen.BNo11 = dlarray(zeros(2*gf,1,'single'));
paramsGen.BNs11 = dlarray(ones(2*gf,1,'single'));
paramsGen.TCW6 = dlarray(initializeGaussian([4,4,gf,4*gf]));
paramsGen.TCb6 = dlarray(zeros(gf,1,'single'));
paramsGen.BNo12 = dlarray(zeros(gf,1,'single'));
paramsGen.BNs12 = dlarray(ones(gf,1,'single'));
paramsGen.TCW7 = dlarray(initializeGaussian([...
    4,4,settings.image_size(3),2*gf]));
paramsGen.TCb7 = dlarray(zeros(settings.image_size(3),1,'single'));

exState = 'stGen.BN%d = [];';
for i = 1:12
    eval(sprintf(exState,i));
end
end

function [paramsDis,stDis] = InitializeDis(settings,df)
paramsDis.CNW1 = dlarray(initializeGaussian([4,4,settings.image_size(3)+1,df]));
paramsDis.CNb1 = dlarray(zeros(df,1,'single'));
paramsDis.CNW2 = dlarray(initializeGaussian([4,4,df,2*df]));
paramsDis.CNb2 = dlarray(zeros(2*df,1,'single'));
paramsDis.BNo1 = dlarray(zeros(2*df,1,'single'));
paramsDis.BNs1 = dlarray(ones(2*df,1,'single'));
paramsDis.CNW3 = dlarray(initializeGaussian([4,4,2*df,4*df]));
paramsDis.CNb3 = dlarray(zeros(4*df,1,'single'));
paramsDis.BNo2 = dlarray(zeros(4*df,1,'single'));
paramsDis.BNs2 = dlarray(ones(4*df,1,'single'));
paramsDis.CNW4 = dlarray(initializeGaussian([4,4,4*df,8*df]));
paramsDis.CNb4 = dlarray(zeros(8*df,1,'single'));
paramsDis.BNo3 = dlarray(zeros(8*df,1,'single'));
paramsDis.BNs3 = dlarray(ones(8*df,1,'single'));
paramsDis.CNW5 = dlarray(initializeGaussian([4,4,8*df,1]));
paramsDis.CNb5 = dlarray(zeros(1,1,'single'));

exState = 'stDis.BN%d = [];';
for i = 1:3
    eval(sprintf(exState,i));
end
end