
# Paper Name:
**_Doing the Best We Can with What We Have:_**
**_Multi-Label Balancing with Selective Learning for Attribute Prediction_**
# publishing information
HAND, E.; CASTILLO, C.; CHELLAPPA, R.. Doing the Best We Can With What We Have: Multi-Label Balancing With Selective Learning for Attribute Prediction. AAAI Conference on Artificial Intelligence

# 1. background problem/motivation:
* CelebA contains a variety of very significant biases, so it is difficult to generalized a model trained in this dataset for use on another dataset.
* The typical approache to dealing with imbalanced data involves sampling the data in order to balance the positive and negative labels, however, with a mulit-label problem this becomes a non-trivial task.
* 2016 Moon propose a mixed objective loss, which adjusts the back-propagation weights according to a given target distribution.
* does not known what real distribution is, so this is a form of domain adapative.

# 2. the proposed methods:
* propose a novel selective sample method
* propose a multi-task attribute CNN
# 3. dataset:
dataset | AttCNN+SL| MOON + SL| 
------------ | -------------|--------------
CelebA | 91.05| 86.33
LFWA | 73.03| 70.49 
UMD-AED |71.11 | 59.46 

# 4. advantages:

# 5. the detail of methods:
* AttCNN
the architecture as follow:
![](/images/DBWCWH-AttCNN-architecture.jpg)
FC3 use sigmoid function
preprocessing steps: subtract the training mean, take a random crop, starting with random initialization
* selective learning
it adaptively balances each label according to the desired distribution for each label in a multi-task learning framework.there are three cases;
  1. the batch distribution is equal to the target distribution, do nothing.
  * the label is over-represented, which means there are more positive instances than negative. Taking following steps: take a random subset from positive samples  according to the target distribution and add to the SL batch, ignoring the rest. Second, weight the negative samples  so they effectively match the target distribution.
  * the label is under-represented, reverse above process.

# 6. contribution:

# 7. any questions during the reading :

# 8. vocabulary:
celebrities 名人
under-represented 样例太少
bald 秃头
Lipstick 口红
curves 曲线
versatile 多才多艺
metric 公正
feat 功绩

# 9 中文理解
## Introduction

CelebA 主要由正面、高清的名人照片组成，存在各种类型的偏差，无法泛化、泛化太差，是仅有的数据集 其他数据集都太少了，所以只能克服这个偏差 为此提出了一个在每一个batch 里根据每一个label想要的分布自动平衡数据的神经网络选择性学习方法，**是否已经解决bias？**

Moon 这篇论文提出了一个目标损失函数，本文作者觉得很棒，就接着往下做，提出了创新点，既然你提出了每个example都有一个权重，我想到神经网络不是每个batch 每个batch地训练，那我就提出一个在每一个batch都使用的 domain-adaptive batch re-sampling method 称为selective learning，它根据对每个属性给定的目标分布为每一个属性分别调整每个batch

## Related work
### Attributes
属性很重要，也研究了很久介绍了脸部属性识别的相关工作
### Domain Adaptation
介绍了一大堆监督半监督的域适应问题的解决方法，然后说脸部识别可以被认为是面部适应问题，面部具有不同姿态与不同照明不同分辨率，半监督学习被拿来做这个方面。
介绍了MOON 怎么做的，改进了啥，
Facial attribute representation learning from egocen- tric video and contextual data。是现在最好的算法
在多个数据集 CelebA、LFWA、UMD-AED 效果都还行，对每个batch 实行多标签平衡

监督域适应，已知道$X_s, X_T, Y_s,Y_T$
无监督域适应，不知道$Y_T$

## proposed method
### multi-task attribute CNN
使用AttCNN 来识别属性，结构图如下:
![](/images/DBWCWH-AttCNN-architecture.jpg)
* F3 是输出层，有40个节点，每一个代表一个属性，使用的是sigmoid，
### selective learning
对训练数据采取多标签平衡方法，选择性学习通过根据该标签的目标分布自适应地平衡每批数据中的每个标签，为此问题提供解决方案。
对每个属性学习一个模型就很简单，就可以从更平衡的数据集中学习，但是在多任务条件下，一次学习所有数据，处理数据不平衡很困难， 
#### batch balancing

batch 的情况分为三种，一种是正好合适，这时候不用进行处理， 第二是 正例多于反例，这时候根据目标分布对正例进行随机采样，忽略掉剩下部分的正例，然后对反例进行加权处理， 这时候的batch 就是包含正例的一部分，以及全部的反例，以及一个而外的权重。第三种就是把第二种反过来处理
那目标分布怎么来？
#### implement
对于一个属性a，我们有目标分布$P_T(a)$ 以及batch 分布$P_B(a)$
如果$P_T(a) = P_B(a)$ 那么，它的loss就正常计算
如果$P_B(a=1) > P_T(a=1)$，正例太多，其中$P_T(a=1)$权重设为1，其他设为0，反例就乘以$\frac{P_T(a=0)}{P_B(a=0)}$
在MOON里，没考虑到每个batch，只考虑了全局的目标分布，这会导致训练的不平衡
然后通过图展示了有selective learning 与没有的区别
介绍了说这个方法可以对任何多标签方法都适用，具有很大的创新点

## experiments
数据集: 
CelebA 都是名人的高清正面照，包括20万张，16万训练、2万验证与测试，每张照片有40个属性
LFWA 只有13k照片，跟CelebA很想
UMD-AED 它自己弄的一个验证数据集，2.8k照片每个属性都有50个正例与反例，它的变化很多
在CelebA测试集上效果比其他方法都好

### AttCNN
减掉图片平均，再随机裁剪，从256\*256到227\*227,权重随机初始化，没有预训练，表现还比三种方法好，参数量也少只有6million
认为AttCNN的成功归功于直接对属性数据进行训练，参数少
？？？ 效果怎么这么好的，是否使用了selective learning
### selective learning
第一个测试
使用AttCNN带上select learning 使用平衡的目标分布
？？？目标分布？？？
训练了22轮，batch size 200 在CelebA上表现没有MOON好

第二个测试
使AttCNN 训练更适应CelebA训练分布以及CelebA测试分布，提高了效果。
证明bias在测试集里仍然存在

目标分布会很大影响测试结果

认为label 贴的不行，有些属性太主观了，
口红这个属性有太多噪音，贴的太差啦
下一步应收集一个更大规模更准确的数据集

## conclusion
最主要的思想就是提出了 在每个batch 里面给 反向传播带入权重
提出了一个AttCNN，效果很好

## 疑惑
基于名人的训练集，它是怎么 truly represent the facial attributes 的
为什么可以有时候知道$Y_T$
什么是 domain adaptation

如何提高泛化误差
如何设置更为合适的目标分布
仅仅使用一个AttNet 效果就这么好了？
接下来该怎么做
仍有什么缺陷？
