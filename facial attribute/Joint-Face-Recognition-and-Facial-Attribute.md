
# Paper Name:
**_Multi-task Deep Neural Network for Joint Face Recognition and Facial Attribute Prediction**

# publishing information
Z. Wang, K. He, Y. Fu, R. Feng, Y.-G. Jiang, and X. Xue.
Multi-task Deep Neural Network for Joint Face Recognition
and Facial Attribute Prediction. ICMR, 2017.
# 1. background problem/motivation:

# 2. the proposed methods:

# 3. dataset:

# 4. advantages:

# 5. the detail of methods:

# 6. contribution:

# 7. any questions during the reading :

# 8. vocabulary:
biometric 生物识别
trait 特征
facilitate 促进
renaissance 复兴
distractor 干扰
scenario 场景
prohibitive 望而却步
orthogonal 正交
and vice versa 反之亦然
cue 线索
off-the-shelf 现成的
intrinsically 本质
prune 修剪
outlying 边远

# 中文翻译
## abstract
提出一个多任务的深层网络来学习一起学习脸部识别与脸部属性预测两个任务。
实验证明，在MegaFace这个数据集上，效果很好，达到了77.74%

## introduction
脸部识别很用，应用广泛，现在深度学习取得了很大的进展，特别是在LFW 这个数据集上，但是对于MegaFace 数据集，效果不行，它是拿来模拟更为真实的应用场景，并作为benchmark，标准的解决方案是，更深的网络或者更多的训练数据，这花费太大， 于是，可能属性有帮助，就引入脸部属性来共同训练。
本文提出联合学习两种任务的方法，使用多数投票的方法对数据集贴标签，使用额外的数据集进行预训练，比如CelebA，使用的是MOON架构。提出的架构学习了脸部识别与脸部属性分类两种任务。实验证明效果很好。
总结，主要有三点贡献，第一，第一篇在这个领域第一篇联合学习的文章，相比最先进的方法有很大的进步。
第二，还提出了fast版本，第三，在直观上与效果上系统性地探索了不同架构与配置的替代选择

## related work
### face recognition analysis
简单介绍了脸部发现与脸部对准，在介绍了脸部验证与脸部识别的不同，
face verification and face identification

### face attribute analysis
最近有很大突破，相关工作可以分为三类，一种是sift 传统方法， 二是 基于深度特征的脸部属性分析，比如 [Liu et al.](Deep-Learning-Face-Attributes-in-Wild.md)
三是multi-task框架下的子任务。比如[Rudd et al.](MOON.md)

### multi-task learning
多任务学习是一种迁移学习，它解决了学习共享信息和利用同一数据集上不同任务之间相似性的问题。

## the attribute-constrained deep face recognition architecture
其他方法没有直接使用脸部属性

### architecture overview
分为两步，一步是脸部属性预测，第二步是脸部识别网络
第一步是使用现成的脸部属性预测方法对数据集进行预测，
第二步是将图片作为输入送入网络进行预测

### facial attribute prediction
仅仅使用了一部分CelebA数据集中的属性，不是全部，使用CelebA训练MOON，然后用来预测CAISIA-WebFace 数据集，一个实例有多张照片，所以要使用 投票机制，对实例进行判断，

仅仅使用了CelebA一部分的属性，而不是全部的，而且使用了已经存在的MOON方法， 并对其进行优化。
