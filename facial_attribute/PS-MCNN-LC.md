
# Paper Name:
**_Partially Shared Multi-Task Convolutional Neural Network with Local Constraint for Face Attribute Learning_**
# publishing information
JiajiongCao,YingmingLi and ZhongfeiZhang.  Partially Shared Multi-Task Convolutional Neural Network with Local Constraint for Face Attribute Learning. 2018,CVPR.

# 1. background problem/motivation:

* it needs intensive experiments to find the optimal split point, especially for deep networks; 
* The shared information vanishing among groups emerges when it reaches the high level layers of MCNN.
* more importantly, interactions among different tasks at high-level layers are restricted since there are no shared lay- ers after the bifurcation.

# 2. the proposed methods:
* propose a novel Partially Shared Multi-task Convolutional Neural Network (PS-MCNN) in which task relation is captured by a Shared Network (SNet) and variability across different tasks is captured by Task Specific Networks (TSNets)

* incorporate identity information into PS-MCNN to improve the performance of multi-task face attribute learning

* introduce local learning constraint which minimizes the difference between the representations of each sample and its local geometric neighbours with the same identity

# 3. dataset:
CelebA/LFWA

# 4. advantages:
* identity information help boost the performance of the net.
* partially solved the interaction problem between different groups
# 5. the detail of methods:

# 6. contribution:
* firstly introducte a PS-MCNN and an LCLoss 
# 7. any questions during the reading :

# 8. vocabulary:
complementary  互补
bifurcation 分支
fraction 比例
feature fusion 融合

# 中文翻译  
## Abstract
本文同时考虑 身份信息与属性关系，提出 部分共享的多任务卷积网络(PS-MCNN)，包含四个特定任务的网络(TSNets),以及一个分享网络(SNet),为了更好使用身份信息，引入局部学习限制(local constraint)，最小化 每个样本表示与具有相同身份的局部几何邻域之间的差异。提出的方法就叫做 PS-MCNN-LC
## Introduction
介绍脸部属性的应用，同时存在问题， 由于有了CNN的方法，提升很大， 现在最新的方法是 MCNN ，介绍了MCNN的优点， 同时又介绍了 MCNN的不足，
不同组之间的相互作用受到限制，因为它们在分裂后是独立的。 当信息到达MCNN的高层时，群组中共享的信息就会消失。 因此，属性组很难有效地利用从网络开始到结束的属性相关性来提升整体性能。

本文就是解决了组之间的交互问题，同时还利用了身份信息。

我们假定，结合身份信息提高准确度，这个假定是基于 不同属性组之间有效的交互会帮助得到更准确的属性关系模型，
具有信息的身份标签通过对属性学习模拟局部的几何机构来大大提升效果。

首先，提出PS-MCNN SNet先抓取任务的关系，TSNet 抓取不同任务的变化，与MCNN相似的是，所有属性分为几个组，然后每个组的分类学习可以被认为是独立的任务。不同的是 不同的组会分享特征。
直接引入身份信息不可取，所以就通过局部限制损失来引入

## Partially Shared Network for Face Attribute Learning

### Split structure
多任务学习。的通常方法，就跟MCNN一样，指出了MCNN的不足，是 需要大量的实验来找到最优分开属性的方案，第二是，不同任务在高层次的layer内没办法交互，在分开之后就没办法分享信息了

### Partially shared structure
为了克服这个问题，提出PS-MCNN，包含两个网络 TSNet、SNet，TSNet专注于学习特定任务的信息，SNet 是用来学习每个任务都可以分享的信息表示，计算方式，点加

### Partially shared multi-task convolutional neural network
将40个属性按方位分为4个区域，分别代指脸部四个不同区域

### Design decisions

需要进一步研究设置网络。
网络初始化问题，SNet与TSNet 的分享比例，

## Partially Shared Network with Local Constraint for Face Attribute Learning
考虑如何将identity information加入网络的方法。
### Local learning constraint
介绍 其他方法如何使用identity information，同一个 identity 的属性应该尽可能相近，所以提出了一个LCLoss， 
### Partially shared multi-task CNN with local constraint
将LCLoss 加入 PS-MCNN的loss 中。由一个$\lambda$ 控制 LCLoss 的比例。


## Ablative Analysis
本文使用了预训练，TSNet 、SNet 分别在 CelebA 上的face recognition task 与 attribute prediction task 中 预训练。
网络没有共享浅层的网络。

这个number of channels of SNet  需要在考虑一下是啥意思。

### Effectiveness of PS Architecture
训练了4个相互独立group的网络，与PS-MCNN进行对比。效果很明显。

### Complexity Analysis
拿MCNN 对比了PS-MCNN的参数，对比比较了两个网络之间的区别。


## Experiment
使用CelebA对齐过的数据进行训练，一开始只用identity LOss 训练 SNet，后面移除了FC层， 将其与TSNet 连接。

比较之前的方法

说明LCLoss 的有效性

## Conclusion
