
# Paper Name:
**_Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift_**
# publishing information
S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep
network training by reducing internal covariate shift. In ICML, 2015.[[paper]](https://arxiv.org/abs/1502.03167)
# 1. background problem:
  * the distribution of each layer's inputs changes during training, as the parameters of the previous layers change.
  * the training is complicated by the fact that the inputs to each layer are affected by the parameters of all preceeding layers -so that small changes to the network parameters amplify as the network becomes deeper.
  * whitened the layer inputs is expensive, as it requires computing the covariance matrix  and its inverse square root,to produce the whitened activations 

# 2. the proposed methods:
  * propose a new mechanism, which called Batch Normalization, that takes a step towards reducing internal covariate shift. 

# 3. dataset:

# 4. advantages:

# 5. the detail of methods:
  * Normalization via Mini-Batch Statistics
    * normalize each scalar feacture independently,by making it have the mean of zero and the variance of one. Namely, norlalize each dimension:
    $$ \hat{x}^{(k)} = \frac{x^{(k)} - E\left [x^{(k)}  \right ] }{\sqrt{Var\left [ x^{(k)} \right ]}}$$

    * introduce a pair of parameters $ \gamma^{(k)} $ and $ \beta^{(k)}$, which scale and shift the normalized value: 
    $$ y^{k} = \gamma^{(k)} * \hat{x}^{(k)} + \beta^{(k)} $$

    * since we use mini-batch in stochastic gradient learning, **each mini-batch produces estimates of the mean and variance of each activation**
# 6. contribution:

# 7. any questions during the reading :
  * what is the **_internal convariate shift_**
  answer: it is the change in the distribution of network activations due to the change in network parameters during training.

# 8. vocabulary:
notoriously hard 出了名
exceed 超过
proceed 继续
oppose 反对
preceding 前面
notion 概念
regime 政权
substantial 坚实的
interval 间隔
interspersed 穿插
whitened 白化
subsequent 随后
consequence 结果
scale 规模、比例
scalar 标量
empirically 经验
stand-alone 独立
account for 考虑