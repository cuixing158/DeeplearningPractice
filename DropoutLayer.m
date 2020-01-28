function [out_features,mask] = DropoutLayer(in_features, ratio)
% 功能：以ratio的比例丢弃神经元
% 输入：in_features,输入特征
%       ratio 丢弃比例，[0,1]之间
% 输出：out_features，输出特征,大小与in_features一致
%       mask，随机掩码，大小与in_features一致
% 参考：https://zhuanlan.zhihu.com/p/38200980
%      https://blog.csdn.net/oBrightLamp/article/details/84105097
%
% author:cuixingxing 2020.1.27
% email:cuixingxing150@gmail.com
%

all_nums = numel(in_features);
drop_nums = round(all_nums*ratio);
idxs = randperm(all_nums,drop_nums);
mask = ones(size(in_features));
mask(idxs) = 0;

out_features = in_features.*mask./(1-ratio);
end
