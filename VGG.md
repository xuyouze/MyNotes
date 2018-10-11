
# Paper Name:
**_VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION_**
# 1.background problem:
  * the accuracy of ILSVRC is still low. In order to improve that, this paper modify the architecture of conventional CNN

# 2. the proposed methods:
  * a thorough evaluation of networks of increasing depth using an architecture with 3\*3 convolution filters.
  *  replace 7\*7 conv layers with three 3\*3 conv layers

# 3. dataset:
  * ILSVRC-2012 concludes 1000 classes and training set 1.3M, validation 50K, testing 100k. 
  best top-1 error 23.7% top-5 error 6.8%

# 4. advantages:
  * the first paper of thorough evaluation about the depth on CNN.
# 5. the detail of methods:
  * the detail of the architecture:
    * input: 224*224 RGB IMAGE
    * pre-process: subtracting the mean RGB image value, computed on the training set.
    * max-pooling is using 2\*2 pixel window, with stride 2.
    * convolution stride is fixed to 1 pixel.
    * architecture as follow:

        ![](https://raw.githubusercontent.com/xuyouze/MyNotes/master/images/VGG-structure.jpg)<br/>

  * the advance of using a stack of three 3\*3 conv layers instead of a single 7\*7 layer:
    * it can make the decision more discriminative because of three nonlinear rectification layers
    * it can decrease of the numbers of parameters which can be seen as imposing a regularization on the 7\*7 conv
  * detail of training
    * batch size is 256.
    * momentum is 0.9
    * L2 penalty is 5\* 10^-4.
    * dropout is 0.5
    * learning rate is initially set to 10^-2, and then decreased by a factor of 10
  * the different between dense evaluation and multi-crop evaluation.
    * multi-crops perform slightly better than dense evaluation (0.2%)
    * multi-crop evaluation is complementary to dense evaluation due to different convolution boundary conditions: when applying a ConvNet to a crop, the convolved feature maps are padded with zeros, while in the case of dense evaluation the padding for the same crop naturally comes from the neighboring parts of an image (due to both the convolutions and spatial pooling), which substantially increases the overall network receptive field, so more context is captured.

# 6. contribution:
  * show that Local Response Normalization normalization does not improve the performance on the ILSVRC dataset, but lead to increased memory consumption and computation time

  * indicate that while the additional 1\*1 does help, it is also important to capture spatial context by using conv filter with non-trivial receptive such as 3\*3.

  * confirm that training set augmentation by scale jittering is indeed helpful for capturing multi-scale image statistics

# 7. any questions during the reading:
  * what is dense evaluation?
    * the fully-connected layers are first converted to convolutional layers. the resulting fully-convolutional net is then applied to the whole image.
    * then the result is a class score map with the number of channels equal to the number of classes, and a variable spatial resolution, depend on the input image size.
    * finally, to obtain a fixed-size vector of class score for the image, the class score map is spatially averaged(sum-pooled)

# 8. vocabulary:
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
proportion 比例
injected 注射
decomposition 分解
incorporation 掺入
converge 收敛
substantially 基本上
for reference 以供参考
synchronous 同步
asynchronous 异步
conceptually 概念
non-trivial  不平凡
fusion 聚合

