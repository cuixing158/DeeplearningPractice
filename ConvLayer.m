function Z = ConvLayer(imgs, W)
% 功能：对图像imgs进行卷积操作,W为卷积核
% 输入：imgs 待卷积的图像或特征，存储顺序为 img_height*img_width*img_channel*batchSize,
%            四维数组，float,[0,1]范围;
%      W 为卷积核,大小为 kernel_height*hernel_width*kernel_channel*out_filters,
%         四维数组，float类型,kernel_channel必须与img_channel相等才可以卷积!
% 输出：
%      Z为卷积后的特征图，维度大小为(img_height-kernel_height+1)*(img_width-hernl_width+1)*out_filters*batchSize
%        四维数组，float类型;
% 注意：此为图像卷积操作，边界无填充，支持多维卷积操作
%
%  author:cuixingxing 2020.1.25
% email:cuixingxing150@gmail.com
%

[img_height, img_width, img_channel,batchSize] = size(imgs);
[kernel_height, hernl_width, kernel_channel, out_filters] = size(W);
assert(kernel_channel==img_channel);

Z = zeros(img_height-kernel_height+1,img_width-hernl_width+1,out_filters,batchSize);
for i = 1:batchSize
    for j = 1:out_filters
        W = rot90(W,2);% 用convn函数做卷积需要把卷积核旋转180度,只转第一维，第二维度
        Z(:,:,j,i) = convn(imgs(:,:,:,i),W(:,:,:,j),'valid');
    end
end
end
