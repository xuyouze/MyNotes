
# Paper Name:
**_Heterogeneous Face Attribute Estimation: A Deep Multi-Task Learning Approach_**

# publishing information

# 1. background problem/motivation:

# 2. the proposed methods:

# 3. dataset:

# 4. advantages:

# 5. the detail of methods:

# 6. contribution:

# 7. any questions during the reading :

# 8. vocabulary:
surveillance 监控
retrieval 检索
explicitly 明确的
heterogeneous 异构的
ordinal 序列
nominal 名义
holistic 整体
demographic 人口统计
portray 描绘
cue
salient 突出
mustache 胡子
proprietary 专有的
interval 间隔
intrinsic 固有
compact  凝练的


# 中文翻译
## Abstract
人脸属性有很多应用场景，近几年很多方法都被提出来了，但大多数在特征表示学习期间都没有明确地考虑属性之间的关系与异构(序列vs名义、整体vs局部)，本文，引入一个深度多任务学习方法从单个人脸中联合估计多个异质的属性。
在这个方法里，我们使用CNN 来解决 属性之间的异质与关系，包括对于所有属性共享特征学习，以及对异构属性进行特定类特征学习。
引入LFW+，这是对LFW对扩充，实验表明，提出的效果很好。

## Introduction
属性有多种类别    
1. demographic attribute
1. descriptive visual attribute
之前的工作都是对每个属性分别训练一个模型。，也有方法是多属性一起的，但是存在不足。
属性之间可能有positive 或negative 的关系，不同的属性在数据类型、尺寸，语义上可能是异构的，同时有一些属性像年龄 头发长度是序列的，性别、种族描述是名义上的，这两类特征在数据类型与规模是异构的。
这种属性关系与异构是需要考虑的。
商业系统的算法与数据集是不可能得到的，从一张图片中鲁棒性地估计大量异构的属性是很难的。

## Proposed method
提出了DMTL(Deep Multi-Task Learning)来从一张图中联合估计多个异构的属性，这个方法是从最近的方法中汲取灵感的，且考虑了属性关系与属性的异构性，这个方法包含了 针对所有属性的早期共享特征学习，紧接的是针对异构属性类别的特定类别特征学习。
共享的特征学习自然地利用任务之间的关系来达到鲁棒、有区分性的特征表示。
特定类别的特征学习旨在微调共享特征来达到对每个异构属性类别的最佳估计，
根据已给的有效的共享特征学习与特定类别的特征学习，DMTL 达到了预期效果，且保持低计算量，令它具有很大价值
主要贡献
它是一个有效的多任务学习方法，目的是对大量属性进行联合估计。
在一个网络中，对属性关系与属性异构性进行建模
探究了DMTL在不同数据集测试场景下的泛化能力，
一些之前的工作在之前的论文已经写过了。
本质的提升是 
在特定类别特征学习的扩展，为了解决在数据尺寸类型、语义方面的属性异构。
额外的技术、实现细节
使用6个不同的数据集来评估，以及与state-of-the-art 方法进行比较

## Related Work
### Multi-attribute Estimation From Face

介绍了之前的工作

### Multi-Task Learning in Deep Networks

NN 很适合MTL ，本文提出的方法是在MTL的基础上，但是有一些不同：
本文focus on 从一张人脸中联合估计多个属性
本文目的是提高脸部属性估计的精确度，通过利用属性之间的关系和处理属性的异构性
本文提出的方法是end-to-end 的
考虑了异构属性估计，单属性估计和跨数据库测试的许多实际场景。

## PROPOSED APPROACH
### Deep Multi-task Learning
CNN 可以利用 属性之间的关系
介绍了 传统多任务学习的损失函数。
对其进行了修改， 将共享参数与 类特定的参数 分别表示，

### Heterogeneous Face Attribute Estimation
属性的异构性，之前没有被考虑到，是因为 之前的数据集大部分都是单个或几个属性，没有必要。 许多发表的方法选择对每个单独的属性学习一个模型，。
我们将每个异构类分开，但是每个类都共享特征提取的参数
并将损失函数修改了一下。
每个属性分类 都是靠 先验知识定的

介绍了 名词性、序数性的属性的损失函数
名词性就是交叉墒，序数性就是 均方误差
将属性按 整体、部分、名词性、序数性分为四类。

### Implementation Details
使用AlexNet 来提取共享特征，中间插入了 BN层，对于每个类，都有两个 FC层，
输入  使用 value, category 来 表示 label
训练，使用SGD， 随机初始化参数


## EXPERIMENTAL RESULTS
### Databases
MOROH II
CelebA 
LFWA
FotW
LFW+

### Experimental Setting
对图片进行人脸检测与标志位检测，然后将脸的图片缩放为 256\*256\*3，使用CASIA-WebFace 数据集进行预训练，然后使用不同的数据集进行 fine-tune，
使用AlexNet 与GoogLeNet进行测试，发现两个 效果差不多，然后 AlexNet 更快，所以用AlexNet

### Binary Face Attributes
对于二值性的属性就 只可能是  全局或局部 异构的，而不可能是序数、名义上的异构。
所以，对于CelebA 以及 LFWA 在局部、全局的子网络中使用同样的loss，特别的，使用了一个全局名义上的子网 + 7个局部 名义上 的子网。每个子网里面的属性都是手动分的， 这个我觉得不行。
因为 CelebA全都是 二值的属性。

取得了不错的效果。
