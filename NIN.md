
# Paper Name:

  **_Network In Networks_**

# 1.有什么问题:
~~~

  conventional CNN uses linear filter followed by a nonlinear activation function to scan the input, however the level of abstraction is low with GLM.

  We need to replace GLM with a more potent nonlinear function approximator to enhance the abstraction ability of the local model.

  it is difficult to intepret how the categoriy level information from the objective cost layer is passed back to the previous convolution layer due to the fully connected layer which act as a black box in between

~~~

# 2.提出了什么:

1. propose a structure called mlpconv layer consisting of multiple fully connected layers with nonlinear activation functions

* replace **GLM** with a *'micro network'* structure which is a general nonlinear function approximator.

* propose large deep CNN to enhance model discriminability for local pathes within the receptive field (*增强在感知领域下的局部补丁的模型区分性* ) which is stacking of multiple mlpconv layers.

* relace fully connected layers with ** global average pooling layer **

# 3.数据集:
	ImageNet and

# 4.解决了什么:
~~~
相比于传统NN, CNN 更有效果, 高清图片也可能使模型过拟合,所以就提出了dropout方法来防止过拟合, 使用gpu 使得训练加速
达到了top-1 和top-5 的错误比例更低，37.5%与17%
~~~
# 5.方法的细节:
1. multiple perceptron convolutional layers:
  * why choose MLP as the universal function approximator:
    *  MLP is compatible with the structure of CNN which is trained using bp
    *  MLP can be a deep model itself, which is consistent with the spirit of feature re-use.
  *


# 6.方法有什么贡献:
* NIN is proposed from a more general perspective, the micro network is integrated into CNN structure in persuit of better abstraction for all levels of features.

# 7. 阅读中思考的问题:

1. why filter is a GLM for the underlying data path:

  answer:
* why global average pooling is a structural regularizer

  answer:

* what is the distributions of the latent concepts:

  answer:

* what is cascaded cross channel parametric pooling:

* 1*1卷积 有什么用? (需要再理解一下)
~~~
  1.实现跨通道的交互和信息整合
  2.进行卷积核通道数的降维和升维
~~~
* NIN 与1*1 卷积之间的关系:

  MLP层 可以通过多层1*1卷积层替代

# 8.单词:

discriminability 可区分性<br/>
patch 补丁<br/>
A followed by B  A在B前,随后<br/>
instantiate 实例化 <br/>
perceptron 感知机<br/>
potent 有力的
utilize 利用
prone 倾向
underly 基础的
GLM Generalized Linear Model
extent 程度
latent 潜在
i.e. 这就是说
implicitly 含蓄地
manifold 多种
multilayer perceptron MLP
confidence value 置信度
adopting 使用
over-complete 过度完整

Namely 换句话说
impose 强加
piecewise 分段
convex 凸
endows 赋予
necessarily hold 一定成立
compatible 兼容
