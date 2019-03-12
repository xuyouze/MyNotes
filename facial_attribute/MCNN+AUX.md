
# Paper Name:

**_Attributes for Improved Attributes: A Multi-Task Network
Utilizing Implicit and Explicit Relationships for Facial Attribute Classification_**

# publishing information

# 1. background problem/motivation:

# 2. the proposed methods:

# 3. dataset:

# 4. advantages:

# 5. the detail of methods:

# 6. contribution:

# 7. any questions during the reading :

# 8. vocabulary:
impostor 欺骗者
pairwise 配对

# 中文翻译
## Abstract
属性很重要，以前 属性被认为是独立的，其实并不是，所以需要用多标签学习。这可以利用属性之间的关系来提高分类的效果，本文提出来一个 DMTL + AUX 网络，使用三种方法来分享属性之间的关系， 意识 分享底层的layers、 2是通过讲MCNN的属性分数喂给 AUX 来找到 score-level 的关系，3 是将属性分为N个群，进行分群训练。
相对于每个属性单独训练，效果好多了，而且参数也少。


## Introduction
属性很重要，属性的重要性，属性的用处，介绍属性分类的进展，之前是单独训练没有考虑关系， 然而，有一些例子证明，属性之间是有关系的， 提出自己的方法，又抄了一遍摘要。
主要的贡献。
提出 多任务 深度网络
提出多加一个 AUX 额外的网络来学习关系，
结合MCNN和 AUX 来 利用潜在、直接的属性关系。
多个测试集上效果很好。取得了state of the art 的效果，没有pre-training、alignment 
有效地减少了参数的个数，
## Relate work
介绍 CNN 、介绍 多任务学习、 介绍 属性之前的工作，
## Multi-Task CNN
总体介绍提出的网络
### Architecture
设计的网络  
input -> conv1(7\*7\*75) ->Relu -> MaxPooling(3\*3)  -> Normalization(5\*5) -> conv2(5\*5\*200)  ->Relu -> MaxPooling(3\*3) ->Normalization(5\*5) 然后分为六个子网络
-> conv1(3\*3\*300) ->Relu -> MaxPooling(3\*3) ->Normalization(5\*5) 然后分为9个群
-> FC(512) ->Relu -> dropout -> FC(512)->Relu -> dropout 
手动选择，根据实验来的
使用 sigmoid 函数
只有 随机裁剪、平均，没有其他预处理方法。如果用 每个属性单独训练一个模型，参数就特别多。

### MCNN-AUX
训练完MCNN后，加入一个 FC层，，输入的是MCNN的属性分数，输出的是 每个
AUX就是加了一层FC层，

## Experiment
### Data
CelebA
LFWA

### Independent CNNs
参数太多训练太久

### MCNN
batch size 100，22个epoch， 2.5个小时训练CelebA，1个小时LFWA

### Results
independent CNNs 效果比Liu 好，
为效果不好解释了一下，虽然我们的效果没有特别出众，但是我们没有更多的预训练，

## Conclusion
信息分享很重要，提出的方法减少大量的参数，效果达到了领先水平，属性之间的关系 可以多尝试。




先介绍  这个领域
这个领域存在的问题
数据不平衡
多标签问题

当前的方法

