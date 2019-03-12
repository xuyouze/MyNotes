# svm

`超平面公式` 
$w^Tx +b = 0$

`任意点到超平面距离`
$r = \frac{|w^Tx +b|}{||w||}$ 
推导过程
首先 对任意$x_0$, 有$x_0$到超平面S上的投影 $x_1$, 那么  $w^Tx_1+b = 0 $
$x = x_0 + r * \frac{w}{||w||}$
两边同时乘以 $w^T$
$ w^Tx = w^Tx_0 + r*\frac{w^Tw}{||w||} $
$w^Tw = ||w||^2 $
所以 $ r = \frac{f(x)}{||w||}$

`svm基本模型`
目标函数为$\underset{w,b}{max} (\hat{r})$
满足 $y_i(w^Tx_i + b) = \hat{r}_i >= \hat{r}$
令$\hat{r}$ = 1 也就是最大化
$\underset{w,b}{max} \frac{2}{||w||}$  
$s.t. y_i(w^Tx_i + b) = \hat{r}_i >= 1$

同时可以重写为  $\underset{w,b}{min} \frac{1}{2}||w||^2$
$s.t. y_i(w^Tx_i + b) = \hat{r}_i >= 1$

`对偶问题`         #没看懂
可以通过拉式乘数法将目标函数转换为一个统一的函数
$L(w,b,\alpha) = \frac{1}{2}||w||^2 - \sum^m_{i=1} \alpha_i(1-y_i(w^Tx_i +b))$

令$\theta(w,b) = \underset{a_i>=o}{max}L(w,b,a)$
当每个约束条件都满足的时候，后面那个部分必定为正值，则最优值$\frac{1}{2}||w||^2$
则$\underset{w,b}{min}\theta(w) = \underset{w,b}{min}\underset{a_i >=0}{max}L(w,b,a) = p^* $
将其转换为对偶问题，交换最大最小的位置，
$\underset{w,b}{max}\underset{a_i >=0}{min}L(w,b,a) = d^* $

对$w,b$求偏导为0
得
$w = \sum^m_{i=1} \alpha_iy_ix_i$
$\sum^m_{i=1} \alpha_iy_i = 0$

将以上两项带入统一函数， 可得问题
$ L(w,b, \alpha) = \underset{\alpha}{max}\sum^m_{i=1} \alpha_i - \frac{1}{2}\sum^m_{i=1}\sum^m_{j=1}\alpha_i\alpha_jy_iy_jx^T_ix_j$
约束为$s.t. \sum^m_{i=1}\alpha_iy_i = 0 $ 
且$\alpha_i >= 0 i=1,2...m$


求出$\alpha$后，求$w,b$ 即可得模型
$f(x) = w^Tx +b  = \sum^m_{i=1}\alpha_iy_ix_i^Tx +b$

`KKT 条件`
$\alpha_i >= 0$
$y_if(x_i) -1 >= 0$
$\alpha_i(y_if(x_i) -1) >= 0$
所以$\alpha_i = 0$ 或$y_if(x_i) = 1$
$\alpha_i = 0$ 代表不在边界上
$y_if(x_i) = 1$ 代表在边界上



`SMO算法`

`核函数`
为应对训练样本线性不可分的问题，
通过核函数将样本从输入空间映射到更高维的特征空间中，

$\phi(x)$可表示为 x 映射后的特征向量。 
模型可表示为
$f(x) = w^T\phi(x) + b$

由于高维对偶问题中
$\underset{\alpha}{max}\sum^m_{i=1} \alpha_i - \frac{1}{2}\sum^m_{i=1}\sum^m_{j=1}\alpha_i\alpha_jy_iy_j\phi(x_i)^T\phi(x_j)$

约束为$s.t. \sum^m_{i=1}\alpha_iy_i = 0 $ 
$\phi(x_i)^T\phi(x_j)$ 计算通常是很困难的,所以提出了核函数

$\kappa (x_i,x_j) = <\phi_(x_i),\phi_(x_j)> = \phi(x_i)^T\phi(x_j)$

如果有一种方式可以在特征空间中直接计算内积〈φ(xi · φ(x)〉，就像在原始输入点的函数中一样，就有可能将两个步骤融合到一起建立一个非线性的学习器，这样直接计算法的方法称为核函数方法：

`松弛变量处理outliers方法`







