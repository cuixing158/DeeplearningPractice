
%% 1、数据MNISTData预处理
imageFileNameTrain = 'G:\MNIST\train-images.idx3-ubyte';
labelFileNameTtrain = 'G:\MNIST\train-labels.idx1-ubyte';
nClasses = 10;
order = 0:9;

[X_Train,Label_Train] = processMNISTdata(imageFileNameTrain,labelFileNameTtrain);
Label_Train_h = onehot(Label_Train,nClasses,order);

montage(X_Train(:,:,:,1:9))
title(['Ground Truth:',num2str(Label_Train(1:9)')]);

%% 2、超参数设定
alpha=0.01;
beta =0.01;
ratio = 0.01;
epoch = 2;
batchSize = 10;

W1=1e-2*randn(9,9,1,20);% 第一层为CNN卷积，9*9大小，输入通道为1，输出通道设定为20
W2=(2*rand(95,2000)-1)/20;% 第二层为BP卷积层，设定神经元个数为95，2000为第一层特征融合计算得出的
W3=(2*rand(45,95)-1)/10; % 第三层为BP卷积层，设定神经元个数为45
W4=(2*rand(10,45)-1)/5; % 第四层为BP卷积层，设定神经元个数为10，因为直接与10个数字对应
mmt1 = zeros(size(W1));% 带动量的W1梯度，稳定训练
mmt2 = zeros(size(W2));
mmt3 = zeros(size(W3));
mmt4 = zeros(size(W4));

%% 3、反向传播算法+梯度下降算法，迭代更新参数寻优
numCorrect = 0;% 累计预测正确的样本个数
numAll = 0;% 累计输入网络的样本个数
accuracy = 0;
for i = 1:epoch
    [~, ~, ~,numsImgs] = size(X_Train);
    for    idx= 1:batchSize:numsImgs
        %% batch data
        batchInds = idx:min(idx+batchSize-1,numsImgs);
        batchX = X_Train(:,:,:,batchInds);% 28*28*batchSize
        batchY = Label_Train_h(:,batchInds);% 10*batchSize
        [~,~,~,bs] = size(batchX);
        % show images and true labels
        % montage(batchX);
        % title(['Ground Truth:',num2str(Label_Train(batchInds)')]);
        
        %% 第一层：CNN卷积+relu+池化层
        Z1= ConvLayer(batchX,W1);
        Y1=ReLULayer(Z1);
        A1 =PoolLayer(Y1);
        
        %% 第二层：BP全连接层 ＋relu+ DropoutLayer
        y1=reshape(A1,[],batchSize);
        Z2=W2*y1;
        Y2=ReLULayer(Z2);
        [A2,mask2] = DropoutLayer(Y2, ratio);
        
        %% 第三层：BP全连接层 ＋ relu+ DropoutLayer
        Z3=W3*A2;
        Y3=ReLULayer(Z3);
        [A3,mask3] =  DropoutLayer(Y3, ratio);
        
        %% 第四层：BP全连接层+softmax
        Z4 = W4*A3;
        P = SoftmaxLayer(Z4); % 10*batchSize 大小
        % 计算训练集当前准确率
        [~,ind] = max(P);
        predict_L = onehot(ind-1,nClasses,order);
        for idx_img = 1:bs
            isEqual = predict_L(:,idx_img)==batchY(:,idx_img);
            numCorrect = numCorrect+all(isEqual);
        end
        numAll = numAll+bs;
        accuracy = numCorrect/numAll;
        fprintf('第%d epoch，第%d/%d代总体训练集准确率为：%.2f\n',i,floor(idx/batchSize),floor(numsImgs/batchSize),accuracy);
        
        %% 递推求误差
        % 求第四层误差，即最后一层误差
        e4 = batchY-P;
        
        % 求第三层误差
        delta4 = e4;
        e3=W4'*delta4;
        delta3=mask3.*e3;% DropoutLayer求导 https://blog.csdn.net/oBrightLamp/article/details/84105097
        delta3 = (Z3>0).*delta3;% Relu求导
        % 求第二层误差
        e2=W3'*delta3;
        delta2=mask2.*e2;
        delta2=(Z2>0).*delta2;
        % 求第一层误差
        e1=W2'*delta2;
        e1 = reshape(e1,size(A1));
        avg_e = e1/4; % avg pool层求导参考 https://blog.csdn.net/qq_21190081/article/details/72871704
        E1 = repelem(avg_e,2,2);
        delta1=(Z1>0).*E1;
        
        % 求取梯度并更新W1,按照CNN卷积层的误差反向传播原理https://www.cnblogs.com/pinard/p/6494810.html
        [~,~,~,numFilters]=size(W1);
        for idx_f =1:numFilters
            dW1 = zeros(size(W1));% 9*9*1*20
            for idx_img = 1:bs
                dW1(:,:,:,idx_img)=alpha* convn(batchX(:,:,:,idx_img),rot90(delta1(:,:,idx_f,idx_img),2),'valid');
            end
            dW1 = mean(dW1,4); % batch 梯度方向均值
            S = size(W1);S(end) = 1;
            dW1= reshape(dW1,S);% 保持维度
            mmt1(:,:,:,idx_f)= dW1 + beta*mmt1(:,:,:,idx_f);
            W1(:,:,:,idx_f)=W1(:,:,:,idx_f) + mmt1(:,:,:,idx_f); % 带动量形式的mini-batch梯度下降算法
        end
        % 求取梯度并更新W2,按照BP反向传播原理
        dW2=alpha*delta2*y1';% 因为第一层是CNN，卷积后的特征是全连接，故y1是A1的变换形式
        mmt2 = dW2 + beta*mmt2;
        W2   = W2 + mmt2;
        % 求取梯度并更新W3,按照BP反向传播原理
        dW3=alpha*delta3*A2';
        mmt3 = dW3 + beta*mmt3;
        W3   = W3 + mmt3;
        % 求取梯度并更新W4,按照BP反向传播原理
        dW4=alpha*delta4*A3';
        mmt4 = dW4 + beta*mmt4;
        W4   = W4 + mmt4;
        
    end
    % 每个epoch完后进行保存
    save(['model_epoch',num2str(i),'.mat'], 'W1','W2','W3','W4');  
end



