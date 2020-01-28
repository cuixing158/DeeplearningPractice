function out = preProcess(in)
% 功能：预处理输入图像
% 输入：in 为H*W*numImgs 大小，uint8类型图像
% 输出：out 为H*W*C*numImgs大小，float类型[0,1]范围图像,mnist数字图像，其中C=1
%
[H,W,numImgs] = size(in);
out = im2single(in);
out = reshape(out,[H,W,1,numImgs]);
end