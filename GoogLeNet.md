
# Paper Name:
**_Going deeper with convolution_**
# 1.有什么问题:
  * to improve the performance of cnn in accuary and speed .
  * under the limit of size of the training set , bigger size means a larger number of parameters,which makes the enlarged network more prone to overfitting
  * another drawback of uniformly increased network size is the dramatically increased use of computational resources.

# 2.提出了什么:
  * propose a deep convolution neural network architeture codenamed Inception
  * propose a cnn architeture which is stacking of Inception

# 3.数据集:

# 4.有什么缺陷:

# 5.方法的细节:
  * why was the Inception proposed ?
    * answer

  * what the idea of Inception?
    * It is based on finding out how an optimal local sparse structure in a convolutional vision network can be approximated and covered by readily available dense conponents.

  * what is the beneficial of the Inception?
    * it allows for increasing the number of units at each stage significantly without an uncontrolled blow-up in computational complexity.The ubiquitous use of dimension reduction allows for shielding the large number of input filters of the last stage to the next layer, first reducing their dimension before convolving over them with a large patch size.

    *  it aligns with the intuition that visual information should be processed at various scales and then aggregated so that the next stage can abstract features from different scales simultaneously.

  * the structure of Inception as followd:

  ![Aaron Swartz](https://raw.githubusercontent.com/xuyouze/MyNotes/master/images/Inception.jpg)
    * 1*1 convolution                     
    * 1\*1 convolution + 3\*3 convolution
    * 1\*1 convolution + 5\*5 convolution
    * 3\*3 maxpooling + 1\*1 convolution  
    it is a combination of all those layers with their output filter banks concatenated into a single output vector forming the input of the next stages.


# 6.方法有什么贡献:

# 7.阅读中不懂的问题:
  * what is Hebbian principle?

    answer:
  * what is Gabor filters?

    answer:
  * what is the purpose of using 1*1 convolutions:
    * use as a dimension reduction modules to remove computational bottlenecks
    * width network without significant performance penalty

  * what is this means:
  ~~~
    The fundamental way of solving both issues would be by ultimately moving
    from fully connected to sparsely connected architectures, even inside the convolutions.
  ~~~
# 8.单词:
hallmark 标志
incarnation 实例
synergy 协同效应
sheer 纯粹
inference 推理
conjunction 连词
contrast 对比
contrary 相反
bottlenecks 瓶颈
decomposes 分解
cue 线索
agnostic 不可知
tricky 棘手的
fine-grained 细粒度
uniformly 均匀
quadratic 二次
prefer 更倾向于
mimicking 模仿
firmer theoretical underpinnings 更坚实的理论基础
resonate 共鸣
underlying 底层
infrastructures 基础设施
vast 巨大
speculative 投机
undertaking 承诺、事业
modest 谦逊
bound to vary 必然会变化
prohibitively expensive 非常昂贵
pronounced 明显
judiciously 明智地
projections 投影
embedding 嵌入
aggregated 合计
depict 描绘
ubiquitous 普及
