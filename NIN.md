
# Paper Name:

  **_Network In Network_**

# 1.有什么问题:

  * conventional CNN uses linear filter followed by a nonlinear activation function to scan the input, however the level of abstraction is low with GLM.

  * We need to replace GLM with a more potent nonlinear function approximator to enhance the abstraction ability of the local model.

  * it is difficult to intepret how the categoriy level information from the objective cost layer is passed back to the previous convolution layer due to the fully connected layer which act as a black box in between

  * the fully connected layers are prone to overfitting without the dropout, hampering the regularization ability of the overall structure.

# 2.提出了什么:

1. propose a structure called mlpconv layer consisting of multiple fully connected layers with nonlinear activation functions
* replace **GLM** with a **'micro network'** structure which is a general nonlinear function approximator.

* propose large deep CNN to enhance model discriminability for local pathes within the receptive field (*增强在感知领域下的局部补丁的模型区分性* ) which is stacking of multiple mlpconv layers.

* replace **fully connected layers** with **global average pooling layer**

# 3.数据集:
	CIFAR-10, CIFAR-100, SVHN, MNIST

# 4.解决了什么:
~~~

~~~
# 5.方法的细节:
1. why choose MLP as the universal function approximator:
  *  MLP is compatible with the structure of CNN which is trained using bp
  *  MLP can be a deep model itself, which is consistent with the spirit of feature re-use.
* why replace fully connected layer with global average pooling :
  * the  fully connected layers are prone to overfitting, which hampering the generalization ability of the overall work.
  * it is more native to the convolution structure by enforcing correspondences between feature maps and categories
  * there is no parameters to optimize in the global average pooling



# 6.方法有什么贡献:
  * propose mlp conv layer and 1*1 kernel convolution layer.
  * replace fully connected layer with global average pooling layer

# 7. 阅读中思考的问题:

1. why filter is a GLM for the underlying data path:

  answer:
* what is the distributions of the latent concepts:

  answer:
* what is cascaded cross channel parametric pooling:

  就是mlp conv layer. 也等价于 1*1 的卷积层

* 1*1卷积 有什么用? (需要再理解一下)
  * 实现跨通道的交互和信息整合
  * 进行卷积核通道数的降维和升维

* what is data augmentation:

  用来解决数据量过少对问题， 对原数据进行偏移、裁剪 等操作。

* what is the different between global average pooling and fully connected layers:
  * 都对feature map 进行了矢量化特征映射的线性变换, 但是变换矩阵不同
  * 对于 *GAP*, 它是 前缀的，只有在相同值上的对角元素上不为零。
  * 对于 *FC*,  它可以拥有密集的变换矩阵,并且值可以通过bp 调整。

# 8.单词:

discriminability 可区分性<br/>
patch 补丁<br/>
A followed by B  A在B前,随后<br/>
instantiate 实例化 <br/>
perceptron 感知机<br/>
potent 有力的<br/>
utilize 利用<br/>
prone 倾向<br/>
underly 基础的<br/>
GLM Generalized Linear Model<br/>
extent 程度<br/>
latent 潜在<br/>
i.e. 这就是说<br/>
implicitly 含蓄地<br/>
manifold 多种<br/>
multilayer perceptron MLP<br/>
confidence value 置信度<br/>
adopting 使用<br/>
over-complete 超完备<br/>
cross channel parametric pooling layer 交叉通道参数池化层<br/>
Namely 换句话说<br/>
impose 强加<br/>
piecewise 分段<br/>
convex 凸<br/>
endows 赋予<br/>
necessarily hold 一定成立<br/>
compatible 兼容<br/>
hamper 妨碍<br/>
surpasses 超过<br/>
benchmark 基准<br/>
