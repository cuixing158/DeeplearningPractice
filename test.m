%% 测试集测试
load model_epoch2.mat
imageFileNameTest = 'G:\MNIST\t10k-images.idx3-ubyte';
labelFileNameTest = 'G:\MNIST\t10k-labels.idx1-ubyte';
nClasses = 10;
order = 0:9;
% 预处理
[X_Test,Label_Test] = processMNISTdata(imageFileNameTest,labelFileNameTest);
Label_true = onehot(Label_Test,nClasses,order);% 10*numImgs
% 随意查看前16个数字情况
montage(X_Test(:,:,:,1:16))
title(['Ground Truth:',num2str(Label_Test(1:16)')]);

%% 所有样本整体准确度
[~,~,~,numImgs] = size(X_Test);
predict_L=Predict(W1,W2,W3,W4,X_Test);
numCorrect = 0;
for idx_img = 1:numImgs
    isEqual = predict_L(:,idx_img)==Label_true(:,idx_img);
    numCorrect = numCorrect+all(isEqual);
end
numAll = numImgs;
acc = numCorrect/numAll;

fprintf('测试集整体准确率为：%.5f\n',acc);