
# Paper Name:
**_Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification_**

# publishing information
He, K., Zhang, X., Ren, S., and Sun, J. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. ArXiv e-prints, February 2015.
[[paper]](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
# 1. background problem:

# 2. the proposed methods:
  * propose a new generalization of ReLU, which can adaptively learn the parameters of the rectifiers at negligible extra cost.
  * derive a robust initialization method that particularly considers the rectifier nonlinearities.

# 3. dataset:

# 4. advantages:

# 5. the detail of methods:
  * LReLU
    * when $a_{i}$ is a small and fixed value, PReLU become Leaky ReLU. ($a_{i} = 0.01$)
    * however LReLU has negligible impact on accuracy compared to ReLU.
  
  * PReLU
    * the formula of PReLU:
        $$f(y_{i}) = max(0,y_{i}) + a_{i}min(0,y_{i}) $$
    * $a_{i}$ is a learnable parameter for every channel. 
    $y_{i}$ is the input of the nonlinear activation **_f_**
    * do not use weight decay(L2 regularization) when updating $a_{i}$, and init value is 0.25.

# 6. contribution:

# 7. any questions during the reading :
  * what is the Xavier?<br>
    W = np.random.randn(in, out) / sqrt(in) 
# 8. vocabulary:
delve 钻研
expedite 促进
negligible 微不足道
from scratch 从头开始
slope 坡度
mutually 相互
symmetric 对称
magnitude 大小




