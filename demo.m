clear all;
addpath 'natsortfiles'
%% CNN
net = resnet50();
layout = 'avg_pool';
featuresSize = 2048;
inputSize = [224,224];
%% Train autoencoder
normalIdx = 1;
trainAutoencoder = 0;
if(trainAutoencoder)
    %% Train DataSet
    trainDatasetPath = 'RobotPiCamera_train\';
    trainImds = imageDatastore(fullfile(trainDatasetPath), ...
    "IncludeSubfolders",true,"LabelSource","foldernames");
    trainFeatures = zeros(length(trainImds.Files),featuresSize);
    trainLabels = trainImds.Labels;
    trainImds.Files = natsortfiles(trainImds.Files);
    %% Extract train features
    batch = 512;
    %
    idx = 1:batch:length(trainImds.Files);
    idx = [idx, length(trainImds.Files)];
    for i = 1:length(idx)-1
        j = 1;
        imgs = zeros(inputSize(1),inputSize(2),3,length(idx(i):idx(i+1) - 1));
        for idx2 = idx(i):idx(i+1) - 1
            filePath = trainImds.Files{idx2};
            img = imread(filePath);
            imgs(:,:,:,j) = imresize(img,inputSize);
            j = j + 1;
        end
        feats = activations(net,imgs,layout);
        feats = squeeze(feats);
        trainFeatures(idx(i):idx(i+1) - 1,:) = feats.';
        i/length(idx)
    end
    %% Autoencoder
    option.MaxEpochs = 1000;
    option.hiddenSize = 200;
    option.L2WeightRegularization = 1.0e-10;
    option.SparsityRegularization = 1.0e-10;
    option.SparsityProportion = 0.7;
    option.ScaleData = true;
    option.UseGPU = true;
    %% Train Autoencoder
    encoders = [];
    minmaxValues = [];
    labels = unique(trainLabels);
    subTrainFeatures = trainFeatures(trainLabels == labels(normalIdx),:);
    subLabelFeatures = trainLabels(trainLabels == labels(normalIdx),:);
    % normalize
    tmin = min(min(subTrainFeatures));
    tmax = max(max(subTrainFeatures));
    minmaxValues(normalIdx,:)= [tmin, tmax];
    %
    P = subTrainFeatures;
    P = ((P-minmaxValues(normalIdx,1))/(minmaxValues(normalIdx,2))-minmaxValues(normalIdx,1));
    % train autoencoder
    autoenc = trainAutoencoder(P.', option.hiddenSize, ...
        'MaxEpochs', option.MaxEpochs, ...
        'L2WeightRegularization', option.L2WeightRegularization, ...
        'SparsityRegularization', option.SparsityRegularization, ...
        'SparsityProportion', option.SparsityProportion, ...
        'ScaleData', option.ScaleData, ...
        'UseGPU', option.UseGPU);
    encoders{normalIdx} = autoenc;
else
    load('pirobot_encoders.mat');
end
%% Test DataSet
testDatasetPath = 'DataSet_SEQUENCE_RobotPiCamera_RGB_test1\';
testImds = imageDatastore(fullfile(testDatasetPath), ...
"IncludeSubfolders",true,"LabelSource","foldernames");
testFeatures = zeros(length(testImds.Files),featuresSize);
testLabels = testImds.Labels;
testImds.Files = natsortfiles(testImds.Files);
%% Extract test features
batch = 512;
%
idx = 1:batch:length(testImds.Files);
idx = [idx, length(testImds.Files)];
for i = 1:length(idx)-1
    j = 1;
    imgs = zeros(inputSize(1),inputSize(2),3,length(idx(i):idx(i+1) - 1));
    for idx2 = idx(i):idx(i+1) - 1
        filePath = testImds.Files{idx2};
        img = imread(filePath);
        imgs(:,:,:,j) = imresize(img,inputSize);
        j = j + 1;
    end
    feats = activations(net,imgs,layout);
    feats = squeeze(feats);
    testFeatures(idx(i):idx(i+1) - 1,:) = feats.';
    i/length(idx)
end
%% test Autoencoder 
threshold = 0.08;
anomlayIdx = 2;
labels = unique(testLabels);
P = testFeatures(testLabels == labels(normalIdx) | testLabels == labels(anomlayIdx),:);
labelF = testLabels(testLabels == labels(normalIdx) | testLabels == labels(anomlayIdx),:);
rlabel = labelF == labels(anomlayIdx);
P = ((P-minmaxValues(normalIdx,1))/(minmaxValues(normalIdx,2))-minmaxValues(normalIdx,1));
code = encode(encoders{normalIdx},P.');
decoded = decode(encoders{normalIdx},code);
ra = pdist2(P,decoded.','cosine');
resultsD = diag(ra);
fresultsD = resultsD > threshold; 
%result
sum(fresultsD == rlabel) / length(fresultsD)