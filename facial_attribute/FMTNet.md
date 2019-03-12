
# Paper Name:
Multi-label learning based deep transfer neural network for faicial attribute classification

# publishing information

# 1. background problem/motivation:

# 2. the proposed methods:

# 3. dataset:

# 4. advantages:

# 5. the detail of methods:

# 6. contribution:

# 7. any questions during the reading :

# 8. vocabulary:

# 中文翻译
## Abstract
在实际应用中，标签数据仅用于某些常用属性（如年龄，性别）;而未标记的数据可用于其他属性（例如吸引力，发际线），提出基于面部属性分类的多标签学习的新型深度迁移网络，称为FMTNet，包含三个子网络: Face detection Network, the Multi-label learning Network, Transfer learning Network.
FNet 是基于Faster R-CNN 的，然后对脸部探测进行了优化，MNet 用FNet来优化，目的是为了预测带标记数据的多个属性，同时开发了一个有效的损失权重方案来明确地利用基于属性分组的面部属性之间的相关性。TNet通过利用无监督域自适应来训练未标记的面部属性分类

## Introduction
介绍了早期工作， 将方法分为两类，一类是单标签，一类是多标签，多标签好，数据存在问题， 观察到这个问题，提出了，脸部属性分类方法，实施基于多标签的迁移学习，具体一点，就是利用迁移DNN 来预测在目标分布不存在标签的数据的脸部属性，为了更有效利用在源分布中的的数据，考虑脸部属性之间的关系，使用了多标签学习的方法。
FNet 使用 ImageNet预训练过，然后使用脸部属性预训练，MNet是基于FNet的，使用有标注的数据进行优化的，TNet 是用来在目标域中进行分类的。
主要贡献
在source domain 里使用多标签学习，提出了有效的损失权重方法对基于属性群内的属性之间关系进行明确的利用，大大提高了泛化能力，
基于多标签学习，提出的方法利用迁移学习在目标领域对没有标注的数据进行预测，缓解了对全标注训练数据的依赖。

## Related work
### Deep earning
介绍历年的方法，吹了一波自己的方法
### Multi-label learning
讲了一下multi-label 的相关历史，本文中，我们将多标签学习问题分解为多个二元分类问题。 ？？？ 这合理？
### Transfer learning
transfer learning 是一种根据提供的源域的标注数据，提高分类器在目标域的性能的有效方法，这有效地减少了标注的花费。
transfer learning 也指没有监督的域适应。
## The proposed method
介绍了提出的方法

### FNet
介绍了Fast R-CNN 工作方式， 输入是任意的图像，输出脸部位置。

### MNet
在FNet 参数基础上进行优化，对每个属性训练三个全连接层，
减少后面三层的维度，减少计算复杂度，使用逐标签的形式处理多标签学习，将多标签学习问题转化为多个二分类问题，为了提高性能，还利用了属性之间的关系。
卷积层数据跟FNet一样。 后三层fc 是通过数据fine-tune 的，维度分别为512，256，2，
卷积层是共享的，对于每个属性都有三个fc层，使用softmax-cross-entropy 损失函数，分别相加40个属性。
使用softmax-cross-entropy来训练，

将脸部属性分类考虑为多个二分类问题，同时对每个属性还有一个权重， 我觉得这个权重设计的不够合理，
它是根据群组来设置的， 而群组又是人工设定的，每个属性的 权重 设计方案为 $\frac{1}{G} * \frac{1}{g_m}$, 同个群组内的权重一样，
群组很重要，它是关键，损失分配是根据群组的，同时， TNet 的效果也因为它而提高

### TNet
第三个网络是迁移学习网络，TNet 与MNet 很像，是对MNet的fine-tune得到的，两个数据集分布不同，且一个有标注数据，一个没有标注数据，为了解决这个问题，引入 一个 Reproducing Kernel Hilbert Space (再生性希尔伯特空间)，它是一个高维空间， 将分布映射到这个空间中，进行比较，从而缩小两个分布的差异， 距离定义使用的是 Multi-Kernels Maximum Mean Discrepancies(MK-MMD)
简单介绍一下，迁移学习的这个思路：
源 概率分布 p
目标 概率分布 q
源数据 $D_s = \{(X^s, Y^s)\}$ 
目标数据 $D_t = \{X^t\}$

再生性希尔伯特空间有这么一个特点： 可以用空间中的点积表示 $f-> f(x)$的映射， 即
$$f(x) = <f, \phi(x)>_{\mathcal {H}}$$
有如下推导，
$$d_f(p,q) = MMD[\mathcal{F}, p, q] = \underset{ ||f||_{\mathcal{H} \leq 1}}{sup} E_p[f(\phi (X^s))] - E_q[f(\phi(X^t))] \\ = 
\underset{ ||f||_{ \mathcal{H} \leq 1}}{sup} E_p[<\phi (X^s),f>_{\mathcal{H}}] - E_q[<\phi (X^t),f>_{\mathcal{H}}] \\ = 
\underset{ ||f||_{\mathcal{H} \leq 1}}{sup} <E_p[\phi (X^s)] - E_q[\phi (X^t)], f>_{\mathcal{H}} \\= 
|| E_p[ \phi(X^s)] -E_p[ \phi(X^t)]||_{\mathcal{H_f}} 
$$
使用高斯核
$$\tag{1.1} k(x,x') = exp(-||x-x'||^2/(2\sigma^2)) $$
可得
$$ d^2_k(p,q) = E_{x^s, x^{s'}}[k(x^s, x^{s'})] - 2E_{x^s, x^{t}}[k(x^s, x^{t})] + E_{t^s, x^{t'}}[k(x^t, x^{t'})]$$

也就是
$$\underset{D_s,D_t}{min} D_F(D_s,D_t) = \underset{p,q}{min}d^2_k(p,q)$$
$D_F $是两个域在全连接层之间的域差异

总loss 表示为 
$$L = \sum^3_{i=1}D_{F_i} + \alpha L_s$$
前一项表示MK-MMD loss，六个全连接层的差异，这一项只用到了两个属性的特征，没有用到源域的标签，
后一项表示为源域的 loss

没有用迁移学习直接学习不同分布的属性，会导致效果很差，同个群的可能会效果好一点，但也不一定，所以必须要用迁移学习。

## Experiment 
### Datasets and parameter settings
CelebA 
LFWA
### REsults on the multi-label learning network
MNet 使用CelebA 162,770来训练，使用19,962 测试
MNet 使用LFWA 6,571来训练，使用6,571 测试

#### Influence of loss weight
使用不同的loss 权重方案来测试MNet的效果。证明效果还行。

#### Single-label learning vs multi-label learning 
比较了一下 三种方法的效果。

#### Comparion with the state-of-the-art methods
MNet取得了更好或差不多的 效果。

### Results of the proposed method

在CelebA上测试TNet， LFWA 太不平衡了

#### Direct transfer
在一个属性上进行训练， 然后直接去预测另外一个属性，效果很差。

#### Transfer under different correlations
随机选择8种属性作为 source domain 随机选择8种 作为 target domain 然后分别进行一对一训练。
当同一个群时，效果比较好，如果不是，效果就很差， 

### Comparison with the state of the art methods
实验表明 当不是同个群时，$\alpha$ 设置小一点比较好，当同个群时，应设置大一点
有效的原因， 使用阶级式训练方法，后一个网络都是基于上一个网络进行fine-tune的。
通过利用脸部属性之间的关系、提出一个基于属性群的权重分配方案对 已经存在的 多标签学习以及 迁移学习进行了 提高。
