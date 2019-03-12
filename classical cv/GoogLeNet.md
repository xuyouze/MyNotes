
# Paper Name:
**_Going deeper with convolution_**
# publishing information
C.Szegedy,W.Liu,Y.Jia,P.Sermanet,S.Reed,D.Anguelov,D.Er- han, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.[[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html)
# 1. background problem:
  * to improve the performance of CNN in accuracy and speed.
  * under the limit of size of the training set, bigger size means a larger number of parameters, which makes the enlarged network more prone to overfitting
  * another drawback of uniformly increased network size is the dramatically increased use of computational resources.

# 2. the proposed methods:
  * propose a deep convolution neural network architecture codenamed Inception to approximate the expect optimal sparse structure.
  * propose a CNN architecture named GoogLeNet which is consist of Inception.

# 3. dataset:
  * the same with VGG-net, submit in ILVRC14.

# 4. advantages:
  * Inception
    * it allows for increasing the number of units at each stage significantly without an uncontrolled blow-up in computational complexity. The ubiquitous use of dimension reduction allows for shielding the large number of input filters of the last stage to the next layer, first reducing their dimension before convolving over them with a large patch size.

    * it aligns with the intuition that visual information should be processed at various scales and then aggregated so that the next stage can abstract features from different scales simultaneously.

  
# 5. the detail of methods:
  * why was the Inception proposed?
    * the fundamental way of solving those issues would be by ultimately moving from fully connected to sparsely connected architectures.
    * It started out as a case study of the first author for assessing the hypothetical output of a sophisticated network topology construction algorithm that tries to approximate a sparse structure implied by for vision networks and convering the hypothesized outcome by dense,readily available components 

  * what is the idea of Inception?
    * It is based on finding out how an optimal local sparse structure in a convolutional vision network can be approximated and covered by readily available dense components.

  * see this [note](https://zhuanlan.zhihu.com/p/32702031) may help understand this paper.

  * the structure of Inception as followed:
    * 1\*1 convolution  
    * 1\*1 convolution + 3\*3 convolution
    * 1\*1 convolution + 5\*5 convolution
    * 3\*3 maxpooling + 1\*1 convolution  
    it is a combination of all those layers with their output filter banks concatenated into a single output vector forming the input of the next stages.

      ![Inception structure](../images/Inception.jpg)<br/>

  * the detail of GoogLeNet:
    * input size is 224\*224\*3.
    * all convolution use ReLU activation.
    * include 22 layers when counting only layers with parameters 
    * set momentum 0.9 and fixed learning rate schedule decreasing the rate by 4% every 8 epochs.

# 6. contribution:
  * Use 1x1 convolution to perform lifting dimension
  * apply multi-scale convolution on previous stage and concatenated them into a single output vector forming the input of the next stages
  * improve that moving to sparser architectures is feasible and useful idea in general.
  
# 7. any questions during the reading:
  * what is Hebbian principle?
    answer: 
    Hebbin原理是神经科学上的一个理论，解释了在学习的过程中脑中的神经元所发生的变化，用一句话概括就是fire together, wire together。赫布认为“两个神经元或者神经元系统，如果总是同时兴奋，就会形成一种‘组合’，其中一个神经元的兴奋会促进另一个的兴奋”。比如狗看到肉会流口水，反复刺激后，脑中识别肉的神经元会和掌管唾液分泌的神经元会相互促进，“缠绕”在一起，以后再看到肉就会更快流出口水。用在inception结构中就是要把相关性强的特征汇聚到一起。这有点类似上面的解释2，把1x1，3x3，5x5的特征分开。因为训练收敛的最终目的就是要提取出独立的特征，所以预先把相关性强的特征汇聚，就能起到加速收敛的作用
  * what is Gabor filters?

    answer: 
    It is a linear filter used for texture analysis, which means that it basically analyzes whether there are any specific frequency content in the image in specific directions in a localized region around the point or region of analysis. 

  * what is the purpose of using 1*1 convolutions:
    * use as a dimension reduction module to remove computational bottlenecks
    * width network without significant performance penalty

  * what is this means:
  ~~~
    The fundamental way of solving both issues would be by ultimately moving
    from fully connected to sparsely connected architectures, even inside the convolutions.
    
  ~~~
  * why the computer infrastructures is very inefficient when it comes to numerical calculation on non-uniform sparse data structures.
    answer:
  
# 8. vocabulary:
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
culmination 起点
judiciously 明智地
embedding 嵌入
aggregated 合计
depict 描绘
ubiquitous 普及
oriented 面向
sparsity 稀疏性
imply 隐晦
inferior 劣势
homage 尊敬
ensemble 整体
omit 忽略
marginally 轻微地
minor 次要的
discard 丢弃
extra
extraction
exact
schematic 概要
evenly 均匀地
hierarchy 等级制度
deem 认为
aforementioned 上述
elaborate 阐述
refine 精细
viable 可行的