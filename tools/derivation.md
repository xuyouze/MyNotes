# 矩阵求导


## 标量对矩阵求导的运算规则

### 加减法：
$d(X\pm Y) = dX \pm dY$
###乘法
$d(XY) = (dX)Y + X dY $
###矩阵求导的逆：
$dX^{-1} = -X^{-1}dX X^{-1}$
###迹：
$d\text{tr}(X) = \text{tr}(dX)。$
标量套上迹：$a = \text{tr}(a)$
转置：$\mathrm{tr}(A^T) = \mathrm{tr}(A)$
线性：$\text{tr}(A\pm B) = \text{tr}(A)\pm \text{tr}(B)$
矩阵乘法交换：$\text{tr}(AB) = \text{tr}(BA)$，其中$A$与$B^T$尺寸相同。两侧都等于$\sum_{i,j}A_{ij}B_{ji}$。
### 转置
$d(X^T) = (dX)^T$

### 逐元素
$d(X\odot Y) = dX\odot Y + X\odot dY$，$\odot$表示尺寸相同的矩阵X,Y逐元素相乘。
$d\sigma(X) = \sigma'(X)\odot dX ，\sigma(X) = \left[\sigma(X_{ij})\right]$
是逐元素标量函数运算， $\sigma'(X)=[\sigma'(X_{ij})]$是逐元素求导数。举个例子
$X=\left[\begin{matrix}x_{11} & x_{12} \\ x_{21} & x_{22}\end{matrix}\right], d \sin(X) = \left[\begin{matrix}\cos x_{11} dx_{11} & \cos x_{12} d x_{12}\\ \cos x_{21} d x_{21}& \cos x_{22} dx_{22}\end{matrix}\right] = \cos(X)\odot dX$

$A^T(B\odot C) = (A\odot B)^TC$


### 标量对矩阵联系
tr代表迹(trace)是方阵对角线元素之和
$df = \sum_{i=1}^m \sum_{j=1}^n \frac{\partial f}{\partial X_{ij}}dX_{ij} = \text{tr}\left(\frac{\partial f}{\partial X}^T dX\right) $

###例子
####求 $f = \boldsymbol{a}^T X\boldsymbol{b}，求\frac{\partial f}{\partial X}$
解：先使用矩阵乘法法则求微分，这里的$\boldsymbol{a}$, $\boldsymbol{b}$是常量，$d\boldsymbol{a} = \boldsymbol{0}, d\boldsymbol{b} = \boldsymbol{0}$，得到：$df = \boldsymbol{a}^T dX\boldsymbol{b} $，再套上迹并做矩阵乘法交换：$df = \text{tr}(\boldsymbol{a}^TdX\boldsymbol{b}) = \text{tr}(\boldsymbol{b}\boldsymbol{a}^TdX)$，注意这里我们根据$\text{tr}(AB) = \text{tr}(BA)$交换了$\boldsymbol{a}^TdX$与$\boldsymbol{b}$。对照导数与微分的联系$df = \text{tr}\left(\frac{\partial f}{\partial X}^T dX\right)$，得到$\frac{\partial f}{\partial X} = (\boldsymbol{b}\boldsymbol{a}^T)^T= \boldsymbol{a}\boldsymbol{b}^T$。
####例4【线性回归】：$l = \|X\boldsymbol{w}- \boldsymbol{y}\|^2$， 求$\boldsymbol{w}$的最小二乘估计，即求$\frac{\partial l}{\partial \boldsymbol{w}}$的零点。其中$\boldsymbol{y}$是m×1列向量，X是m$\times$ n矩阵，$\boldsymbol{w}$是$n×1$列向量，$l$是标量。
$l = (X\boldsymbol{w}- \boldsymbol{y})^T(X\boldsymbol{w}- \boldsymbol{y})$
$dl = (Xd\boldsymbol{w})^T(X\boldsymbol{w}-\boldsymbol{y})+(X\boldsymbol{w}-\boldsymbol{y})^T(Xd\boldsymbol{w}) = 2(X\boldsymbol{w}-\boldsymbol{y})^TXd\boldsymbol{w}$
根据 $dl = \frac{\partial l}{\partial \boldsymbol{w}}^T d\boldsymbol{w} $ 可得 
$\frac{\partial l}{\partial \boldsymbol{w}}= (2(X\boldsymbol{w}-\boldsymbol{y})^TX)^T = 2X^T(X\boldsymbol{w}-\boldsymbol{y})$

#### 例6【多元logistic回归】：
$l = -\boldsymbol{y}^T\log\text{softmax}(W\boldsymbol{x})$，求$\frac{\partial l}{\partial W}$。其中$\boldsymbol{y}$是除一个元素为1外其它元素为0的m×1列向量，W是m$\times$ n矩阵，$\boldsymbol{x}$是n×1列向量，l是标量；$\text{softmax}(\boldsymbol{a}) = \frac{\exp(\boldsymbol{a})}{\boldsymbol{1}^T\exp(\boldsymbol{a})}$，其中$\exp(\boldsymbol{a}$)表示逐元素求指数，$\boldsymbol{1}$代表全1向量。


## 标量/向量对向量
$df = \sum_{i=1}^n \frac{\partial f}{\partial x_i}dx_i = \frac{\partial f}{\partial \boldsymbol{x}}^T d\boldsymbol{x} $

## 矩阵对矩阵求导
###定义:
$\mathrm{vec}(dF) = \frac{\partial F}{\partial X}^T \mathrm{vec}(dX)$

$\mathrm{vec}(X) = [X_{11}, \ldots, X_{m1}, X_{12}, \ldots, X_{m2}, \ldots, X_{1n}, \ldots, X_{mn}]^T(mn×1)$

标量对矩阵的二阶导数，又称Hessian矩阵，定义为$\nabla^2_X f = \frac{\partial^2 f}{\partial X^2} = \frac{\partial \nabla_X f}{\partial X}(mn×mn)$，是对称矩阵。对向量$\frac{\partial f}{\partial X}$或矩阵$\nabla_X f$求导都可以得到Hessian矩阵

`线性`：$\mathrm{vec}(A+B) = \mathrm{vec}(A) + \mathrm{vec}(B)$。
`矩阵乘法`：$\mathrm{vec}(AXB) = (B^T \otimes A) \mathrm{vec}(X)$，其中$\otimes$表示Kronecker积，A(m×n)与B(p×q)的Kronecker积是A$\otimes$ B = $[A_{ij}B](mp×nq)$。此式证明见张贤达《矩阵分析与应用》第107-108页。
`转置`：$\mathrm{vec}(A^T)$ = $K_{mn}\mathrm{vec}(A)$，A是m×n矩阵，其中$K_{mn}(mn×mn)$是交换矩阵(commutation matrix)。
`逐元素乘法`：$\mathrm{vec}(A\odot X) = \mathrm{diag}(A)\mathrm{vec}(X)$，其中$\mathrm{diag}(A)(mn×mn)$是用A的元素（按列优先）排成的对角阵。

`链式法则`:$\frac{\partial F}{\partial X} = \frac{\partial Y}{\partial X}\frac{\partial F}{\partial Y}$。



## softmax 求导代码

num_example = X.shape[0]
num_class = W.shape[1]
y = np.eye(num_example, num_class)[y.reshape(-1)]

Z = np.dot(X, W)
Z -= np.max(Z)
expZ = np.exp(Z)

y_pred = expZ / np.sum(expZ, axis=1).reshape(num_example, -1)
loss = -np.sum(np.multiply(y, np.log(y_pred)))
loss /= num_example
dW = np.dot(X.T, (y_pred - y))
loss += reg * np.sum(W * W)
dW /= num_example
dW += reg * W


## svm loss
$ L_i = \sum_{j \neq y_i}^{c} max(0,s_j-s_{y_i} + \Delta)= \sum_{j \neq y_i}^{c} max(0,w_jx_i^T-w_{y_i}x_i^T + \Delta)$

当$j\neq y_i, w_jx_i^T-w_{y_i}x_i^T + \Delta > 0$时
$\frac{\partial l_i}{\partial s_j} = x_i^T$

当$j = y_i, w_jx_i^T-w_{y_i}x_i^T + \Delta > 0$时
$\frac{\partial l_i}{\partial s_j} = -x_i^T$

当$w_jx_i^T-w_{y_i}x_i^T + \Delta <= 0$时
$\frac{\partial l_i}{\partial s_j} = 0$
