function predict_L = Predict(W1,W2,W3,W4,X_Test)
% 功能：计算测试集准确率
% 输入：W1,W2,W3,W4为网络权重矩阵
%      X_Test 测试数据，存储顺序为 img_height*img_width*img_channel*numImgs,
%            四维数组，float类型,[0,1]范围;
% 输出：predict_L 预测标签，onehot标签，每列为一个标签
%
%  author:cuixingxing 2020.1.27
% email:cuixingxing150@gmail.com
%

%% 预处理
X = X_Test; % 28*28*1*numImgs， float类型，[0,1]范围
[~,~,~,numImgs] = size(X);

%% forward
%% 第一层：CNN卷积+ReLULayer+池化层
Z1= ConvLayer(X,W1);
Y1=ReLULayer(Z1);
A1 =PoolLayer(Y1);

%% 第二层：全连接层 ＋ReLULayer+ dropout
y1=reshape(A1,[],numImgs);
Z2=W2*y1;
Y2=ReLULayer(Z2);
% [A2,~] = DropoutLayer(Y2, 0.01);

%% 第三层：全连接层 ＋ ReLULayer+ dropout
Z3=W3*Y2;
Y3=ReLULayer(Z3);
% [A3,~] =  DropoutLayer(Y3, 0.01);

%% 第四层：全连接层+softmax
Z4 = W4*Y3;
% 计算训练集准确率
P = SoftmaxLayer(Z4); % 10*batchSize 大小
[~,ind] = max(P);
nClasses = 10;
order = 0:9;
predict_L = onehot(ind-1,nClasses,order);

end
