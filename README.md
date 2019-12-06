# Traffic-flow-forecast 交通流量预测

Use GAT with TA to forecast urban traffic flow.This attempt is to model the crossroad traffic flow and to forecast the unknown crossroad in the roadmap in future.

使用基于时序注意力机制的图注意力网络进行交通流量预测，尝试将交叉路口的车流量进行建模并预测在路网图中未知交叉口的未来的车流量。

## 模型 -- TAGAT： Temporal Pattern - Graph Attention Networks

GAT with TPA , Graph Attention Networks with Temporal Pattern Attention

我们使用GAT对路网结构空间特征进行提取，使用TA对节点的时序车流量数据进行特征提取，将两种时空特征共同作为路网图的特征。GAT使用attention机制提取节点与其邻居的重要性系数，TA使用卷积提取时间特征数据的傅里叶变换，生成feature map，在feature map上使用注意力机制同时关注时间步与时间变化特征pattern对结果的重要性，两种注意力可以同时关注图结构与时序特征，可以显著提升预测任务的准确性。

在数据处理上，使用了时间编码和mask掩码训练方式，将时间序列的周期性和全部训练集特征引入到模型参数中，便于在预测结果时充分考虑交通流量的各个时间分量周期特征和全体路网图特征，从数据处理层面进一步提高预测精度。

## 参考文献

GAT:https://github.com/PetarV-/GAT

TPA:https://github.com/gantheory/TPA-LSTM

## 说明

由于小白我还在撸论文阶段，方向是时间序列预测，这个项目的一部分内容不便于公开，只把主要的框架public出来，项目中目前只提供baseline模型，后期会全部公开，望见谅。



