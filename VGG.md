
# Paper Name:
**_VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION_**
# 1.有什么问题:


# 2.提出了什么:
  1. a thorough evaluation of networks of increasing depth using an architeture with 3*3 convolution filters.
  *  replace 7*7 conv layers with three 3*3 conv layers
# 3.数据集:


# 4.解决了什么:

# 5.方法的细节:
  * the detail of the architeture:
    * input: 224*224 RGB IMAGE
    * pre-process: subtracting the mean RGB image value, computed on the training set

  * the advance of using a stack of three 3*3 conv layers instead of a single 7*7 layer:
    * it can makes the decision more discriminative because of three nonlinear rectification layers
    * it can decrease of the numbers of parameters which can be seen as imposing a regularization on the 7*7 conv

# 6.方法有什么贡献:
  * show that Local Response Normalization normalisation does not improve the performance on the ILSVRC dataset, but lead to increased memory comsumption and computation time

# 7.阅读中不懂的问题:

# 8.单词:
thorough 彻底
prior-art 现有技术
repositories 库
shallow 浅
commodity 商品
densely 密集的
deal with 涉及
brevity 简单的
projection 投影
topology 拓扑结构
crop 作物
instability 不稳定性
circumvent 避免
augment 增强
flip 翻转
conjecture 推测
implicit 含蓄、内在
stall 危害
rescale 重新缩放
jittering 抖动
isotropically 各向同性地
proportion