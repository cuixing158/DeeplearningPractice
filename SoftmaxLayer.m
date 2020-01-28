function y = SoftmaxLayer(x)
%% 功能：softmax转概率
% 输入：x 10*numsSamples大小矩阵
% 输出：y 10*numsSamples概率矩阵，每列为一个样本概率
%
  ex = exp(x);
  y  = ex ./ sum(ex,1);
end