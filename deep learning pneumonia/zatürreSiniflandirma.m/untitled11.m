clear; clc;

% 1. Veri Seti
imgDir = 'chest_xray';
imds = imageDatastore(fullfile(imgDir, 'train'), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% 2. Veri Bölme
[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15);

% 3. Görüntüleri Resize
imageSize = [224 224 3];
augimdsTrain = augmentedImageDatastore(imageSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augimdsValidation = augmentedImageDatastore(imageSize, imdsValidation, 'ColorPreprocessing', 'gray2rgb');
augimdsTest = augmentedImageDatastore(imageSize, imdsTest, 'ColorPreprocessing', 'gray2rgb');

% 4. Katmanlar (ResNet-vari yapıya benzer şekilde daha derin CNN)
numClasses = numel(categories(imdsTrain.Labels));

layers = [
    imageInputLayer([224 224 3],'Name','input')

    convolution2dLayer(7, 64, 'Stride', 2, 'Padding', 'same', 'Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(3,'Stride',2,'Padding','same','Name','pool1')

    convolution2dLayer(3, 64, 'Padding','same','Name','conv2_1')
    batchNormalizationLayer('Name','bn2_1')
    reluLayer('Name','relu2_1')

    convolution2dLayer(3, 64, 'Padding','same','Name','conv2_2')
    batchNormalizationLayer('Name','bn2_2')
    reluLayer('Name','relu2_2')

    maxPooling2dLayer(2,'Stride',2,'Name','pool2')

    convolution2dLayer(3, 128, 'Padding','same','Name','conv3_1')
    batchNormalizationLayer('Name','bn3_1')
    reluLayer('Name','relu3_1')

    convolution2dLayer(3, 128, 'Padding','same','Name','conv3_2')
    batchNormalizationLayer('Name','bn3_2')
    reluLayer('Name','relu3_2')

    averagePooling2dLayer(7, 'Name','avgpool')

    fullyConnectedLayer(256, 'Name','fc1')
    reluLayer('Name','relu_fc1')
    dropoutLayer(0.5, 'Name','dropout1')

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

% 7. Test Et
predictedLabels = classify(netTransfer, augimdsTest);
accuracy = mean(predictedLabels == imdsTest.Labels);

% 8. Kaydet
save('manuel_resnet_benzeri_model.mat', 'netTransfer','accuracy','imdsValidation');
disp("ResNet-benzeri model başarıyla eğitildi ve kaydedildi.");
