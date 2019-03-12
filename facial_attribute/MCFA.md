# Paper Name:
  **_Multi-task learning of cascaded CNN for facial attribute classification PR_**

# publishing information

# 1.background problem/motivation:

* are based on the fixed loss weight without considering the differences of facial attributes
* usually pre-process the input-image(face detection and alignment)
* ignore the inherent dependency of face detection、 facial landmark localization and face attribute classification


# 2. the proposed methods:

**_MFCA_** propose a novel multi-task learning of cascaded cnn method ,
use three cascaded sub-networks (i.e., S_Net, M_Net and L_Net corresponding to the neural networks under different scales) to jointly train multiple tasks in a coarse-to-fine manner the proposed method automatically assigns the loss weight to each facial attribute based on a novel dynamic weighting scheme
# 3. dataset:

# 4.advantages:
* end to end optimization<br/>
* it is the first work to perform multi-task learning in a unified framework for predicting multiple facial attributes simultaneously
* jointly trains different sub-networks in a cascaded manner
* easily apply the back propagation algorithm to train the whole framework.


# 5. the detail of methods:
  网络结构是:
  ~~~
    S-Net
      input: 56*56*3
      architecture: VGG16
      output: 1*256*1*1

    M-Net
      input: 112*112*3
      architecture: VGG16
      output: 1*1280*1*1 ( 1024 + S-Net 256)

    L-Net
      input: 224*224*3
      architecture: VGG-16
      output:  1024 + M-Net 1280)
  ~~~


# 6. contribution:

# 7. any questions during the reading :

# 8. vocabulary:

# 中文翻译
## abstract
之前的方法都忽略了脸部发现、脸部标志性定位和脸部属性预测的关系，而且，这些方法都使用了固定的loss，这不行，所以，提出了级联CNN，同时还设计了动态设置属性loss 的方案，实验证明效果在CelebA以及LFWA上不错

## introduction
介绍应用，然后介绍相关的工作，指出，这些方法都没有考虑到脸部发现、脸部标志与脸部属性的关系， 这忽略了它们之间固有的依赖，然后，对于这些任务，需要有不同的权重。
在这些问题的启发下， 本文提出了MCFA，有三个不同但相关的任务， face detection、facial landmark localization and facial attribute classification 同时被训练，FAC是主任务，其他两个是辅助任务，然后设计了三个级联小网络，S_Net、M_Net、L_Net，在不同尺度提取特征，整个网络以粗到细的方式来执行多任务学习， 集中于对更难的面部属性进行分类。
主要贡献
第一个提出在一个大框架下执行多任务学习以同时对多个脸部属性进行预测，提高了效果。
不同于传统级联网络，本文提出的级联网络是联通的。
提出了动态权重方案，对属性的loss进行平衡。

## propose method
设计了一个三个子网络组成的网络，
S_Net 输入为56\*56\*3,用来提取粗粒度 coarse 的信息，训练的时候，共同训练三个网络，测试的时候只有通过threshold control 层的信息被传到M_Net。

M_Net 输入为112\*112\*3，提取 fine 的特征，训练时，多任务学习，测试时跟上一个网络一样，只有通过才能传到下一个网络。

L_Net 输入为224 \*224\*3, 提取 suble 的特征，训练时，多任务学习，测试时，只有通过 cls 层的信息才能被用来执行多任务学习，
脸部检测包含两个子任务，脸部分类与边框回归，脸部分类是使用softmax ，
边框回归 使用对是 欧式距离， left、top、height、width，为啥用欧式距离， 有测试过吗？

脸部标志点定位，也是欧式距离，
脸部属性分类，认为FAC是一个二分类问题，$u_w\in R^d$ 表示的是对每个属性都有一个权重，
最后的 loss为 
$$L_{joint} = \sum^N_{i=1}\sum^3_{j=1}(L_{ij}^{cls} + L_{ij}^{box}+L_{ij}^{landmark}+L_{ij}^{attr})$$

动态分配loss 给属性。 在S_Net 后 添加了一个conv 层， 在M_Net、L_Net 添加了FC层，

图这样设计有什么用意？


## experiment
介绍了数据集， 需要很多的数据集
将方法拆分，三个任务任意组合， 证明三个任务联合训练很有效
比较了一下别人的