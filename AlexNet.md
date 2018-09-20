
# Paper Name:

   **_ImageNet Classification with Deep Convolutional Neural Networks_**

# 1.有什么问题:
~~~
当前目标识别都是使用 机器学习的方法,数据量太小,目标识别的精度太低,
现在有了大量的数据,我们需要提出一个新的模型来处理。
对LSVRC-2010 contest 的精度太低.
~~~

# 2.提出了什么:
1. propose large, *deep* convolutional neural network, which has 60 million parameters and 650000 neurons,consists of five convolutional layers.
* propose non-saturating neurons **_RELU_**
* use **GPU** to train
* employ a **dropout** method to reduce overfitting in fully connected layers,
* local Response Normalization( **LRN** )

# 3.数据集:
	ImageNet and ILSVRC

# 4.解决了什么:
~~~
相比于传统NN, CNN 更有效果, 高清图片也可能使模型过拟合,所以就提出了dropout方法来防止过拟合, 使用gpu 使得训练加速
达到了top-1 和top-5 的错误比例更低，37.5%与17%
~~~
# 5.方法的细节:
* RELU:

	f(x) = max(0,x), 相同模型, 使用relu 比使用tanh 快好几倍,是为了防止过拟合,解决sigmoid 的梯度弥散问题
* LRU:
	局部响应归一化 增强泛化

# 6.方法有什么贡献:
* we trained one of the largest convolutional neural networks to date on the subsets of ImageNet used in the ILSVRC-2010 and ILSVRC-2012 competitions [2]
and achieved by far the best results ever reported on these datasets.<br/>
* We wrote a highly-optimized GPU implementation of 2D convolution<br/>
* Our network contains a number of new and unusual features which improve its performance and reduce its training time,<br/>
* The size of our network made overfitting a significant problem, even with 1.2 million labeled training examples,<br/>
* we found that removing any convolutional layer (each of which contains no more than 1% of the model’s parameters) resulted in inferior performance.

# 7.单词:

variant 变种<br/>
immense 巨大<br/>
stationarity 固定<br/>
fraction 分数<br/>
