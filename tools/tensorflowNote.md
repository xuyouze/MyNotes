## common
`转换`
tf.cast(x,dtype, name) -> dtype
`维度`
tf.shape(tensor, name) -> tensor or int32
`大小`
tf.size(tensor, name) -> int32
`秩`
tf.rank(tensor,name) -> int32

`reshape 如果shape=[-1] tensor 会被flattened`
tf.reshape(tensor, shape, name) ->tensor

`删除指定维数大小为1 的维数`
tf.squeeze(tensor,axis, squeeze_dim, name) -> tensor

`指定位置插入维度1`
tf.expand_dims(input, dim, name) - >tensor

`切片`
tf.slice(input,begin,size,name) ->tensor

[//]: # value就是输入
tf.split(split_dim, num_split, value, name)->tensor

`复制input multiple 次`
tf.tile(input, multiple, name) -> tensor

`paddings=[[1,2],[3,4]] 就会在 input上面填1行，下面填两行，左边填3行，右边填4行`
tf.pad(input, paddings, name) -> tensor

`调转某一个维度的数据`
tf.reverse(tensor, dims, name ) ->tensor

tf.add(x, y, name=None)
tf.sub(x, y, name=None)
tf.mul(x, y, name=None)
tf.div(x, y, name=None)
tf.mod(x, y, name=None)
tf.add_n(inputs, name=None)
tf.abs(x, name=None)
tf.neg(x, name=None)
tf.sign(x, name=None)
tf.inv(x, name=None)
tf.square(x, name=None)
tf.round(x, name=None)
tf.sqrt(x, name=None)
tf.rsqrt(x, name=None)
tf.pow(x, y, name=None)
tf.exp(x, name=None)
tf.log(x, name=None)
tf.ceil(x, name=None)
tf.floor(x, name=None)
tf.maximum(x, y, name=None)
tf.minimum(x, y, name=None)
tf.cos(x, name=None)
tf.sin(x, name=None)

`对角矩阵`
tf.diag(diagonal, name=None)
`转置，高维比较难懂`
tf.transpose(a, perm=None, name='transpose')
`矩阵乘法`
tf.matmul(a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
`行列式`
tf.matrix_determinant(input, name=None)
`逆矩阵`
tf.matrix_inverse(input, name=None)

`把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解`
tf.cholesky(input, name=None)

tf.reduce_sum(input_tensor, reduction_indices=None, keep_dims=False, name=None)
[//]: # 乘积 沿着 指定的轴。
tf.reduce_prod(input_tensor, reduction_indices=None, keep_dims=False, name=None)
tf.reduce_min(input_tensor, reduction_indices=None, keep_dims=False, name=None)
tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)
tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
tf.reduce_all(input_tensor, reduction_indices=None, keep_dims=False, name=None)

[//]: # Returns the index with the smallest/largest value across dimensions of a tensor.
tf.argmin(input, dimension, name=None)
tf.argmax(input, dimension, name=None)

[//]: # Given a list x and a list y, this operation returns a list out that represents all numbers that are in x but not in y.
[//]: # out: A Tensor. Has the same type as x. 1-D. Values present in x but not in y.
[//]: # idx: A Tensor of type int32. 1-D. Positions of x values preserved in out.
tf.listdiff(x, y, name=None)
[//]: # Returns: A tuple of Tensor objects (y, idx). y: A Tensor. Has the same type as x. 1-D. idx: A Tensor of type int32. 1-D.
tf.unique(x, y,name=None)

[//]: # y[x[i]] = i for i in [0, 1, ..., len(x) -1]
tf.invert_permutation(x, name=None)

## session
tf.Session.__init__(target='', graph=None, config=None)

[//]: # feches 就是需要实现的方法， feed_dict是填入的数据
[//]: # if fetches是方法， 返回null， 如果是 tensor 则返回ndarray
tf.Session.run(fetches, feed_dict=None)

[//]: # 与common session 没多大区别
tf.InteractiveSession()

## value
tf.zeros(shape, dtype, name) -> tensor
tf.ones(shape, dtype,name) -> tensor

[//]: # 与tensor一样的shape 的全为零的tensor
tf.zeros_like(tensor, dtype=None, name=None) -> tensor
tf.ones_like(tensor, dtype=None, name=None) -> tensor

[//]: # dims就是shape, value 为int32或,float32
tf.fill(dims, value, name=None) ->tensor

tf.constant(value, dtype=None, shape=None, name='Const') ->tensor
tf.range(start, limit, delta=1, name='range')
tf.linspace(start, stop, num, name=None)

[//]: # Randomly shuffles a tensor along its first dimension.
tf.random_shuffle(value, seed=None, name=None)
tf.set_random_seed(seed)

## Control Flow Operations:
[//]: # 返回一个一模一样的节点
tf.identity(input, name=None) -> tensor

[//]: # 并行计算tensors
tf.tuple(tensors, name=None, control_inputs=None)

tf.logical_not(x, name=None)
tf.logical_and(x, y, name=None)
tf.logical_or(x, y, name=None)
tf.logical_xor(x, y, name='LogicalXor')
tf.equal(x, y, name=None)
tf.not_equal(x, y, name=None)
tf.less(x, y, name=None)
tf.less_equal(x, y, name=None)
tf.greater(x, y, name=None)
tf.greater_equal(x, y, name=None)

[//]: # output should be taken from t (if true) or e (if false).
[//]: # condition、t、e the same shape, condition is fill of true or false
tf.select(condition, t, e, name=None)                               
[//]: # 不会

[//]: # This operation returns the coordinates of true elements in input.
[//]: # 就是 True 的位置
tf.where(input, name=None)

## framework

tf.Graph.__init__()

[//]: # 2. Constructing and making default:
[//]: # with tf.Graph().as_default() as g:
[//]: #   c = tf.constant(5.0)
tf.Graph.as_default()

[//]: # Finalizes this graph, making it read-only. use for ensure  thread safe
tf.Graph.finalize()

[//]: # with g.control_dependencies([a, b]):
  [//]: # Ops declared here run after `a` and `b`.
[//]: #   with g.control_dependencies([c, d]):
    [//]: # Ops declared here run after `a`, `b`, `c`, and `d`.
tf.Graph.control_dependencies(control_inputs)


[//]: # with g.name_scope("inner"):
[//]: #      nested_inner_1_c = tf.constant(30.0, name="c")
[//]: #      assert nested_inner_1_c.name == "nested/inner_1/c"
tf.Graph.name_scope(name)

tf.Graph.add_to_collection(name, value)
tf.Graph.get_collection(name, scope=None)
tf.Graph.get_operation_by_name(name)
tf.Graph.get_tensor_by_name(name)
tf.Graph.get_operations()

[//]: # a shortcut for calling tf.get_default_session().run(t).
tf.Tensor.eval(feed_dict=None, session=None)

[//]: # get_shape().as_list()
tf.Tensor.get_shape() ->TensorShape
tf.Tensor.set_shape(shape)
tf.TensorShape.as_list()

tf.get_default_graph()

tf.placeholder(dtype, shape=None, name=None) -> tensor

## 读取
class tf.ReaderBase
class tf.TextLineReader
class tf.WholeFileReader
class tf.IdentityReader
class tf.TFRecordReader
class tf.FixedLengthRecordReader

## 激活函数
tf.nn.relu(features, name=None)
tf.nn.relu6(features, name=None)
tf.nn.softplus(features, name=None)
tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
tf.nn.bias_add(value, bias, name=None)
tf.sigmoid(x, name=None)
tf.tanh(x, name=None)

## 卷积
`二维卷积`
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
`深度卷积`
tf.nn.depthwise_conv2d(input, filter, strides, padding, name=None)                            [//]: # 不会
`深度分离卷积`
tf.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, name=None) [//]: # 不会

## 池化
tf.nn.max_pool(value, ksize, strides, padding, name=None)
tf.nn.avg_pool(value, ksize, strides, padding, name=None)
tf.nn.max_pool_with_argmax(input, ksize, strides, padding, Targmax=None, name=None)

## 归一化
`L2范数`
tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)

`LRN 局部响应归一化`
tf.nn.local_response_normalization(input, depth_radius=None, bias=None, alpha=None, beta=None, name=None)[//]: # 不会


`返回平均数跟方差` mean and variance.
tf.nn.moments(x, axes, name=None)


## losses
[//]: # output = sum(t ** 2) / 2 
tf.nn.l2_loss(t, name=None)

## Classification
[//]: # 二分类
[//]: # 它适用于每个类别相互独立但互不排斥的情况,在一张图片中，同时包含多个分类目标（大象和狗），那么就可以使用这个函数。
[//]: # max(x, 0) - x * z + log(1 + exp(-abs(x)))
tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None)

[//]: # Computes softmax activations.
tf.nn.softmax(logits, name=None)

[//]: # 它适用于每个类别相互独立且排斥的情况，一幅图只能属于一类，而不能同时包含一条狗和一只大象
tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)
[//]: # 如果labels的每一行是one-hot表示，也就是只有一个地方为1，其他地方为0，可以使用tf.sparse_softmax_cross_entropy_with_logits()
[//]: # sparse 与非sparse的区别: sparse label输入的是int型 softmax的label 输入的是one-hot
tf.nn.sparse_softmax_cross_entropy_with_logits(sentinel=None,labels=None,logits=None, name=None)

[//]: # SparseTensor(values=[1, 2], indices=[[0, 0], [1, 2]], shape=[3, 4])
[//]: #   [[1, 0, 0, 0]
[//]: #    [0, 0, 2, 0]
[//]: #    [0, 0, 0, 0]]
tf.SparseTensor.__init__(indices, values, shape)

[//]: # 用 default_value 填充 sp_input
tf.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value, name=None)
tf.sparse_tensor_to_dense(sp_input, default_value, name=None)

[//]: # output[d_0, d_1, ..., d_n, sp_input[d_0, d_1, ..., d_n, k]] = True
tf.sparse_to_indicator(sp_input, vocab_size, name=None)


tf.sparse_concat(concat_dim, sp_inputs, name=None)
[//]: # 对内部的排序进行reorder,实际位置并没有变化
tf.sparse_reorder(sp_input, name=None)

[//]: # 对内部元素进行删除, to_retain = [True, False,False, True]
tf.sparse_retain(sp_input, to_retain)

[//]: # 对没有值的行 进行 填充， specified default_value at index [row, 0]
tf.sparse_fill_empty_rows(sp_input, default_value, name=None)


## Variables
tf.Variable.__init__(initial_value, trainable=True, collections=None, validate_shape=True, name=None)
tf.Variable.initialized_value()
tf.get_variable(name, shape=None, dtype=tf.float32, initializer=None, trainable=True, collections=None)
tf.get_variable_scope()
tf.variable_scope(name_or_scope, reuse=None, initializer=None)
tf.constant_initializer(value=0.0)


tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.random_uniform(shape, minval=0.0, maxval=1.0, dtype=tf.float32, seed=None, name=None)
tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None)
tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None)
tf.random_uniform_initializer(minval=0.0, maxval=1.0, seed=None)
tf.uniform_unit_scaling_initializer(factor=1.0, seed=None)
tf.zeros_initializer(shape, dtype=tf.float32)

[//]: # use_locking: If True, use locking during the assignment.
tf.Variable.assign(value, use_locking=False)
tf.Variable.assign_add(delta, use_locking=False)
tf.Variable.assign_sub(delta, use_locking=False)
tf.Variable.scatter_sub(sparse_delta, use_locking=False)
tf.Variable.eval(session=None)
tf.Variable.count_up_to(limit)
tf.Variable.get_shape()

[//]: # Returns all variables collected in the graph.
tf.all_variables()
[//]: # Returns all variables created with trainable=True.
tf.trainable_variables()
[//]: # This is just a shortcut for initialize_variables(all_variables())
tf.initialize_all_variables()
[//]: # Returns an Op to check if variables are initialized.
tf.assert_variables_initialized(var_list=None)

## Saver

[//]: # var_list can be a list of variables or a dict of names to variables:
[//]: # reshape: If True, allows restoring parameters from a checkpoint where the variables have a different shape.
[//]: # sharded: If True, shard the checkpoints, one per device.
[//]: # max_to_keep: maximum number of recent checkpoints to keep. Defaults 5.
[//]: # keep_checkpoint_every_n_hours: How often to keep checkpoints. Defaults to 10,000 hours.

tf.train.Saver.__init__(var_list=None,
                    reshape=False,
                    sharded=False,
                    max_to_keep=5,
                    keep_checkpoint_every_n_hours=10000.0,
                    name=None, restore_sequentially=False, saver_def=None, builder=None)

tf.train.Saver.save(sess, save_path, global_step=None, latest_filename=None)
tf.train.Saver.restore(sess, save_path)
tf.train.Saver.last_checkpoints
tf.train.Saver.set_last_checkpoints(last_checkpoints)
tf.train.update_checkpoint_state(save_dir, model_checkpoint_path, all_model_checkpoint_paths=None, latest_filename=None)


tf.train.Optimizer.__init__(use_locking, name)
tf.train.Optimizer.minimize(loss, global_step=None, var_list=None, gate_gradients=1, name=None)
tf.train.Optimizer.compute_gradients(loss, var_list=None, gate_gradients=1)
tf.train.Optimizer.apply_gradients(grads_and_vars, global_step=None, name=None)



tf.train.GradientDescentOptimizer.__init__(learning_rate, use_locking=False, name='GradientDescent')
tf.train.AdagradOptimizer.__init__(learning_rate, initial_accumulator_value=0.1, use_locking=False, name='Adagrad')
tf.train.MomentumOptimizer.__init__(learning_rate, momentum, use_locking=False, name='Momentum')
tf.train.AdamOptimizer.__init__(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
tf.train.FtrlOptimizer.__init__(learning_rate, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='Ftrl')
tf.train.RMSPropOptimizer.__init__(learning_rate, decay, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp')
tf.stop_gradient(input, name=None)


tf.global_norm(t_list, name=None)
tf.train.Optimizer.get_slot_names()
tf.train.Optimizer.get_slot(var, name)

## Decaying the learning rate

[//]: # global_step = tf.Variable(0, trainable=False)
[//]: # starter_learning_rate = 0.1
[//]: # learning_rate = tf.exponential_decay(starter_learning_rate, global_step,100000, 0.96, staircase=True)
[//]: # optimizer = tf.GradientDescent(learning_rate)
[//]: # Passing global_step to minimize() will increment it at each step.
[//]: # optimizer.minimize(...my loss..., global_step=global_step)
tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)

class tf.train.ExponentialMovingAverage
