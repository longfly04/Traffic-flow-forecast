# Traffic-flow-forecast

Use GCN to forecast urban traffic flow.

## Project Map

step 1: create the graph of roadmap and crossroads;

step 2: load the data and split into temporal slices;

step 3: build the graph network and train the models;

step 4: forecasting, done.

## Model -- TPGAT： Temporal Pattern - Graph Attention Networks

GAT with TPA , Graph Attention Networks with Temporal Pattern Attention

我们使用GAT对路网结构空间特征进行提取，使用TPA对节点的时序车流量数据进行特征提取，将两种时空特征共同作为路网图的特征。GAT使用attention机制提取节点与其邻居的重要性系数，TPA使用卷积提取时间特征数据的傅里叶变换，并在一个训练的batch内对特征按照时间的方向进行计算，这一点与普通的时序attention不同。

训练时，尝试将语言模型中的mask机制应用到图的计算中，随机的mask部分节点的车流数据，并将其作为预测标签，将训练过程转化为有监督训练过程，便于与预测任务有机衔接。基于本次任务，由于预测任务是预测35个交通卡口4天的车流量数据，我们随机mask35个卡口的节点信息，并用全路网的节点训练模型。

GAT:https://github.com/PetarV-/GAT

TPA:https://github.com/gantheory/TPA-LSTM


## TPA temporal pattern attention



