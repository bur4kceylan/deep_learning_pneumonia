clear; clc;

% 1. Veri Seti Yolu
imgDir = 'chest_xray';
imds = imageDatastore(fullfile(imgDir, 'train'), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% 2. Veri Bölme
[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15);

% 3. Resize & RGB
imageSize = [224 224 3];
augimdsTrain = augmentedImageDatastore(imageSize, imdsTrain, 'ColorPreprocessing','gray2rgb');
augimdsValidation = augmentedImageDatastore(imageSize, imdsValidation, 'ColorPreprocessing','gray2rgb');
augimdsTest = augmentedImageDatastore(imageSize, imdsTest, 'ColorPreprocessing','gray2rgb');

% 4. Hafif CNN Modeli (SqueezeNet'e benzer)
numClasses = numel(categories(imdsTrain.Labels));
layers = [
    imageInputLayer([224 224 3],'Name','input')

    convolution2dLayer(3,8,'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')

    convolution2dLayer(3,16,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool2')

    convolution2dLayer(3,32,'Padding','same','Name','conv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    maxPooling2dLayer(2,'Stride',2,'Name','pool3')

    fullyConnectedLayer(64,'Name','fc1')
    reluLayer('Name','relu_fc1')
    dropoutLayer(0.4,'Name','dropout')

    fullyConnectedLayer(numClasses,'Name','fc2')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')
];

% 5. Eğitim Seçenekleri
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% 6. Eğit
netTransfer = trainNetwork(augimdsTrain, layers, options);

% 7. Test Başarı ve Karmaşıklık Matrisi
predictedLabels = classify(netTransfer, augimdsTest);
trueLabels = imdsTest.Labels;

accuracy = mean(predictedLabels == trueLabels);
disp(['Test Doğruluğu: ', num2str(accuracy)]);

confusionchart(trueLabels, predictedLabels);

% 8. Kaydet
save('SqueezeNet.mat','netTransfer','accuracy','imdsValidation');
