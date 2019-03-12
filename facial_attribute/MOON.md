
# Paper Name:
**_MOON: A Mixed Objective Optimization Network for the Recognition of Facial Attributes_**

# publishing information
Rudd, E.; Gunther, M.; and Boult, T. 2016. Moon: A mixed objective optimization network for the recognition of facial attributes. ECCV.
# 1. background problem/motivation:

# 2. the proposed methods:

# 3. dataset:

# 4. advantages:

# 5. the detail of methods:

# 6. contribution:
incorporating domain adaptation into the training procedure for multi-objective attribute classifiers
# 7. any questions during the reading :

# 8. vocabulary:
vision 视觉
albeit 虽然
distilling 提取
mix sth with sth 将什么与什么混合
demographic 人口
binary facial attributes 二进制脸部属性
explicitly 明确地
co-occurrences 共同出现
exogenous 额外的
verification 验证
more or less 或多或少
pioneered 首创
leverage 利用
truncated 截断的
cumbersome 笨重
indirectly 间接
deviates 偏离
radically 根本
contend 认为
heretofore 迄今为止
mitigate 缓解
legibility 易读性
omit 忽略
deem 认为
magnitudes 幅度

# 中文翻译
## abstract
脸部属性识别，一开始是用多个model 来训练，后面发现联合优化更好，展示了对于深度卷积神经网络而言，多任务优化更好，但是由于数据不平衡，所以很难训练，想要使用对数据进行采样来平衡，结果发现会导致其他标签的分布发生变化，所以不可行，
这篇文章就是为了对付这个多标签数据不平衡的问题。引入了一个mixed objecttive optimization network (MOON),该网络具有将多个任务目标与传播损失的域自适应重新加权相混合的损失函数。


## introduction
混合以及联合训练优化对多任务目标有利，多目标学习影响了很多领域。
本文主要解决了脸部属性识别，我们假设它非常适合多目标方法，因为脸部属性具有共享的，虽然是潜在的相关性，但对属性空间有一定的约束
多目标学习没有被大范围使用的可能原因是平衡训练数据的label 很难，之前的方法都是使用two-stage 方法， 训练一个特征提取器然后再对其进行分类，使用Adaboost设定特征空间然后用svm 分类，或使用cnn 提取特征，在用svm分类
本文实验表明联合学习比two-stage学习更好，也比每个属性单独训练的网络更好精度更高、内存更少，不仅更直观也更有效
目标分布与原分布肯定不一样，同时也为了泛化，所以需要domain adaptive，仅仅通过采样输入以及权重是不太合适，太难了，给定目标分布对于单独训练的属性分类器，就可以比较容易地处理，比如在损失函数里面重新加权，所以提出了MOON
有什么贡献呢～
提出的MOON 结构，可以同时学习多个attribute label，只用一个DCNN，还支持
提出了一个公平评价方法，将源分布与目标分布都纳入分类测量从而产生平衡的CelebA评估协议
实验证明同时训练多个label 效果很好
评估了MOON的稳定性

## related work
介绍了一下多目标学习应用在哪些方面，比如目标识别，多个目标可能会在同一张图中，需要对多个目标进行学习，在文本分类问题，对于所有字符联合推理有效，在图像检索标记问题， 面部模型拟合以及特征点估计问题也是一个多任务问题，由于各种面部特征、姿态、光线、表情以及其他外部因素导致的极端差异，就需要微粒度的拟合，同时也受益于全局信息
然后介绍了脸部属性的应用，包括基于语义上的有意思的描述的搜素，以及用人类可以理解的方式解释验证的结果
然后介绍了脸部属性的起源，严重依赖于相对于正脸模版的脸部对齐，每个属性都用的是AdaBoost学习方法，对于每个属性有一个特征，然后再用svm来判断，但是不好。
近年来，方法越来越复杂，介绍了几张方法。主要看Liu 的那篇，是用来三个CNN，两个localization Network，一个attribute recognition Network，功能分别是定位脸部，以及提取脸部特征，然后再输入svm中，进行最后的分类。这个方法是在本文发出前最好的方法，实验表明粗粒度的信息(image level) 的信息可以间接嵌入隐含层 
多任务学习还没被应用到脸部属性识别问题上，只有一篇有，但是不是用CNN来做的，
然后介绍了本文的方法的是第一篇～使用多任务学习来对付这个问题的。

## 方法
方法目的是最大化所有任务的预测准确度
定义了一系列这个问题的方法符号,质疑Liu宣称的，同时优化每个独立的任务，潜在的属性特征会被特征空间表示学习到，训练分布应该跟测试分布一致。
介绍真正的问题，如何合适地平衡用来学习属性特征的数据集。
介绍为啥这个问题值得研究，首先，多任务的数据平衡几乎不可能，其次，训练分布与测试分布不一致。如何解决问题，就是引入一个混合目标函数，对于每个属性都有一个权重。目标分布未知，任意给定，然后把每个属性的损失相加作为总loss
提出了一个详细损失函数。

## 实验
CelebA很大的偏差
第一个实验： 测试没有考虑偏差的情况，也就是源分布等于目标分布，裁剪图片224×224 pixels to 178×218 ，  learning rate of 0.00001， bp方法使用 RMSProp，两种方法，一种分开，一种集成，
介绍了分开训练的方法，分开训练每个属性要训练2个epoch。
集成训练，此时损失函数就是欧式损失。24个epoch，比单个的久，但是比总共的速度快，介绍了平均错误个数。
展示了效果图，非常不错。
还有网络输出，不知道是啥意思
某些属性有很大的提升。
对于平衡的类，得分图就很很平衡，
对于不平衡的类，主导的类别很平衡，但是另一个类就没学好。比如Young,Chubby.
为了得到平衡的得分分布，设置目标分布为0.5、0.5，效果比原来的差因为测试集也有很大的偏差，但是泛化更好了，就需要下篇论文提供的UMD-AED测试集了。
使用了一个新的评价指标：平等错误比例，平衡后的效果更好。

## 讨论
对数据集进行变换，使用MOON进行测试，误差偏差了2%多，但是还是比现在的好，使用变换的数据进行测试，发现错误率降低了，证明MOON具有良好的鲁棒性，
## 结论
MOON效果很好，没用其他数据集就可以再CelebA上效果很好，结合域适应跟多任务目标的混合目标方法，有了很好的效果，
## 疑惑
为啥要域适应
为啥svm 可以很容易对每个标签重平衡训练数据
为什么Liu 那篇不直接使用属性数据来学习特征表示

还能有什么地方可以改进呢
