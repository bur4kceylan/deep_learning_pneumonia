% --- Tek Kodda Eğitim, Doğrulama ve Test ---
clear; clc;

% 1. Veri Seti Yolu
imgDir = 'chest_xray';

% 2. imageDatastore ile Veri Okuma
imds = imageDatastore(fullfile(imgDir, 'train'), ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% 3. Veri Setini Bölme (Eğitim, Doğrulama, Test)
[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15);

% 4. Görüntüleri Yeniden Boyutlandırma ve Ön İşleme
imageSize = [227 227 3];
augimdsTrain = augmentedImageDatastore(imageSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augimdsValidation = augmentedImageDatastore(imageSize, imdsValidation, 'ColorPreprocessing', 'gray2rgb');
augimdsTest = augmentedImageDatastore(imageSize, imdsTest, 'ColorPreprocessing', 'gray2rgb');


% 5. AlexNet Modelini Oluşturma (Ağırlıksız)
layers = [
    imageInputLayer([227 227 3])
    convolution2dLayer(11, 96, 'Stride', 4)
    reluLayer
    crossChannelNormalizationLayer(5)
    maxPooling2dLayer(3, 'Stride', 2)
    groupedConvolution2dLayer(5, 128, 2, 'Padding', 'same')
    reluLayer
    crossChannelNormalizationLayer(5)
    maxPooling2dLayer(3, 'Stride', 2)
    convolution2dLayer(3, 384, 'Padding', 'same')
    reluLayer
    groupedConvolution2dLayer(3, 192, 2, 'Padding', 'same')
    reluLayer
    groupedConvolution2dLayer(3, 128, 2, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(3, 'Stride', 2)
    fullyConnectedLayer(4096)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(4096)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(1000)
    softmaxLayer
    classificationLayer
];

% 6. SeriesNetwork Nesnesi Oluşturma
net = SeriesNetwork(layers);

% 7. Modelin Son Katmanlarını Değiştirme
numClasses = numel(categories(imdsTrain.Labels));
layers(end - 2) = fullyConnectedLayer(numClasses);

% 8. Eğitim Seçenekleri
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% 9. Modeli Eğitme
netTransfer = trainNetwork(augimdsTrain, layers, options);

% 10. Eğitim Sonuçlarını Test Verisi Üzerinden Görüntüleme (eğitim sonuçlarını test ile değerlendiriyoruz ama test kodu en son çalıştırılacak)
predictedLabels = classify(netTransfer, augimdsTest);
accuracy = mean(predictedLabels == imdsTest.Labels);

% 11. Modeli Kaydetme
save('egitilmisZaturreModeli.mat', 'netTransfer','accuracy','imdsValidation');
disp("Model başarıyla eğitildi ve 'egitilmisZaturreModeli.mat' olarak kaydedildi.");

% --- Doğrulama (Validation) İşlemleri ---
try
  % 1. Doğrulama Verisiyle Tahmin Yap
   predictedValidationLabels = classify(netTransfer, augimdsValidation);

    % 2. Doğrulama Performansını Değerlendir
    confusionchart(imdsValidation.Labels, predictedValidationLabels);
    accuracyValidation = mean(predictedValidationLabels == imdsValidation.Labels);
    disp(['Doğrulama Doğruluğu: ', num2str(accuracyValidation)]);

    [precisionValidation, recallValidation, f1ScoreValidation] = calculateMetrics(imdsValidation.Labels, predictedValidationLabels);
    disp(['Doğrulama Hassasiyeti: ', num2str(precisionValidation)]);
    disp(['Doğrulama Geri Çağırması: ', num2str(recallValidation)]);
    disp(['Doğrulama F1 Skoru: ', num2str(f1ScoreValidation)]);

    % 3. Aşırı Uyum (Overfitting) Kontrolü
   disp(['Eğitim Doğruluğu: ', num2str(accuracy)]);
    disp(['Doğrulama Doğruluğu: ', num2str(accuracyValidation)]);
    if accuracyValidation < accuracy
        disp('Aşırı Uyum olabilir, modelde ayarlama yapılması önerilir.');
     else
       disp('Modelin genelleme yeteneği iyi görünüyor.');
   end
catch ME
  error('Doğrulama sırasında bir hata oluştu: %s', ME.message);
end

% --- Test (Test) İşlemleri ---
try
  % 1. Test Verisi ile Tahmin Yap
  predictedTestLabels = classify(netTransfer, augimdsTest);

    % 2. Test Sonuçlarını Değerlendir
    confusionchart(imdsTest.Labels, predictedTestLabels);
    accuracyTest = mean(predictedTestLabels == imdsTest.Labels);
    disp(['Test Doğruluğu: ', num2str(accuracyTest)]);

     [precisionTest, recallTest, f1ScoreTest] = calculateMetrics(imdsTest.Labels, predictedTestLabels);
     disp(['Test Hassasiyeti: ', num2str(precisionTest)]);
    disp(['Test Geri Çağırması: ', num2str(recallTest)]);
    disp(['Test F1 Skoru: ', num2str(f1ScoreTest)]);
catch ME
  error('Test sırasında bir hata oluştu: %s', ME.message);
end

% --- Yardımcı Fonksiyon ---
function [precision, recall, f1Score] = calculateMetrics(trueLabels, predictedLabels)
    % İkili sınıflandırma için performans metriklerini hesaplar
        TP = sum((predictedLabels == 'PNEUMONIA') & (trueLabels == 'PNEUMONIA'));
        FP = sum((predictedLabels == 'PNEUMONIA') & (trueLabels == 'NORMAL'));
       FN = sum((predictedLabels == 'NORMAL') & (trueLabels == 'PNEUMONIA'));

        if (TP + FP) == 0
           precision = NaN;
       else
            precision = TP / (TP + FP);
       end
        
        if (TP + FN) == 0
            recall = NaN;
       else
            recall = TP / (TP + FN);
        end
        if (precision + recall) == 0
          f1Score = NaN;
        else
            f1Score = 2 * (precision * recall) / (precision + recall);
         end
end