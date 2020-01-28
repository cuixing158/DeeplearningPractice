function onehotMatrix = onehot(labels,nClasses,order)
% 产生独热编码矩阵，根据order顺序返回onehot矩阵，其中每列为一个one hot标签
% 输入：labels 包含所有类别标签的类别，1*nums或者nums*1单个类别或者字符向量
%      nClasses 包含类别总数
%      order 长度为numLabels的标签顺序，向量，labels中元素来自于order
% 输出：onehotMatrix
%       独热编码矩阵，nClasses*nums大小，每列中有且仅有一个1，其余为0
% example:
%  labels = [0,3,2,8];
%  nClasses = 10;
%  order = [0,1,2,3,4,5,6,7,8,9];
%  onehotMatrix = onehot(labels,nClasses,order);
%  onehotMatrix = [1  0  0  0
%                  0  0  0  0
%                  0  0  1  0
%                  0  1  0  0
%                  0  0  0  0
%                  0  0  0  0
%                  0  0  0  0
%                  0  0  0  0
%                  0  0  0  1
%                  0  0  0  0]
%
% author:cuixingxing 2020.1.27
% cuixingxing150@gmail.com
%

assert(nClasses==length(order));
assert(all(ismember(labels,order)));
labels = categorical(labels);
order = categorical(order);

E = eye(nClasses);
nums = numel(labels);
indexs = [];
for i = 1:nums
    ind = find(labels(i)==order);
    indexs = [indexs;ind];
end
onehotMatrix = E(:,indexs);
