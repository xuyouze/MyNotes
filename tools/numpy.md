# numpy

`矩阵乘法` **np.dot(A, B)、np.matmul(A,B)** 
`对应元素相乘` **np.multiply(A, B)**、**A\*B** 

`范数、欧式距离` 
**np.linalg.norm** (A, ord=None, axis=None, keepdims=False)
ord:默认是二范数，1 为列和的最大值，，∞ 为行和的最大值
axis:默认是矩阵范数，0 为列向量范数，返回行向量， 1 为行向量范数，返回列向量
keepding:是否保持矩阵的二维特性

### 生成矩阵
`一矩阵` 
**np.ones**(shape,dtype=None, order='C')
shape : (2,3) 或者 3,默认为行向量 
dtype : data-type,The desired data-type for the array, e.g., numpy.int8. Default is numpy.float64.
order : {‘C’, ‘F’}, 行优先还是列优先, default: C
`零矩阵`
**np.ones**(shape,dtype=None, order='C')
`内容一致矩阵`
**np.full**(shape, fill_value, dtype=None, order='C')
`复制大小的矩阵`
**np.ones_like**(a, dtype=None, order='K', subok=True)
**np.zeros_like**(a, dtype=None, order='K', subok=True)
**numpy.full_like**(a, fill_value, dtype=None, order='K', subok=True)

### 计算两个矩阵的欧式距离
$ || A-B||_2 = \sqrt{||X||_2 + ||Y||_2 - 2 * X*Y^T} $

dists = np.sqrt(np.dot(np.sum(np.square(A), axis=1).reshape(num_test, 1), np.ones((1, num_train)))
+np.dot(np.ones((num_test, 1)),np.sum(np.square(B),axis=1).T.reshape(1, num_train)) - 
2 * np.dot(A, B.T))
       