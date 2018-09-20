## common
tf.cast(x,dtype, name) -> dtype

tf.shape(tensor, name) -> tensor or int32

tf.size(tensor, name) -> int32

[//]: # 返回tensor的秩<br/>
tf.rank(tensor,name) -> int32

[//]: # 如果shape=[-1] tensor 会被flattened<br/>
tf.reshape(tensor, shape, name) ->tensor

[//]: # 删除指定维数大小为1 的维数<br/>
tf.squeeze(tensor,axis, squeeze_dim, name) -> tensor

[//]: # 指定位置插入维度1<br/>
tf.expand_dims(input, dim, name) - >tensor

tf.slice(input,begin,size,name) ->tensor

[//]: # value就是输入<br/>
tf.split(split_dim, num_split, value, name)->tensor

[//]: # 复制input multiple 次<br/>
tf.tile(input, multiple, name) -> tensor

[//]: # 例如 paddings=[[1,2],[3,4]] 就会在 input上面填1行，下面填两行，左边填3行，右边填4行<br/>
tf.pad(input, paddings, name) -> tensor

[//]: # 调转某一个维度的数据<br/>
tf.reverse(tensor, dims, name ) ->tensor

[//]: # 转置<br/>
tf.transpose(x) -> tensor

tf.add(x, y, name=None)<br/>
tf.sub(x, y, name=None)<br/>
tf.mul(x, y, name=None)<br/>
tf.div(x, y, name=None)<br/>
tf.mod(x, y, name=None)<br/>
tf.add_n(inputs, name=None)<br/>
tf.abs(x, name=None)<br/>
tf.neg(x, name=None)<br/>
tf.sign(x, name=None)<br/>
tf.inv(x, name=None)<br/>
tf.square(x, name=None)<br/>
tf.round(x, name=None)<br/>
tf.sqrt(x, name=None)<br/>
tf.rsqrt(x, name=None)<br/>
tf.pow(x, y, name=None)<br/>
tf.exp(x, name=None)<br/>
tf.log(x, name=None)<br/>
tf.ceil(x, name=None)<br/>
tf.floor(x, name=None)<br/>
tf.maximum(x, y, name=None)<br/>
tf.minimum(x, y, name=None)<br/>
tf.cos(x, name=None)<br/>
tf.sin(x, name=None)<br/>

[//]: # 对角矩阵<br/>
tf.diag(diagonal, name=None)<br/>
[//]: # 转置，高维比较难懂<br/>
tf.transpose(a, perm=None, name='transpose')<br/>

tf.matmul(a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
[//]: # 行列式<br/>
tf.matrix_determinant(input, name=None)<br/>
[//]: # 逆矩阵<br/>
tf.matrix_inverse(input, name=None)<br/>

[//]: # 把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解<br/>
tf.cholesky(input, name=None)<br/>
tf.reduce_sum(input_tensor, reduction_indices=None, keep_dims=False, name=None)<br/>
[//]: # 乘积 沿着 指定的轴。<br/>
tf.reduce_prod(input_tensor, reduction_indices=None, keep_dims=False, name=None)<br/>
tf.reduce_min(input_tensor, reduction_indices=None, keep_dims=False, name=None)<br/>
tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)<br/>
tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)<br/>
tf.reduce_all(input_tensor, reduction_indices=None, keep_dims=False, name=None)<br/>

[//]: # Returns the index with the smallest/largest value across dimensions of a tensor.<br/>
tf.argmin(input, dimension, name=None)<br/>
tf.argmax(input, dimension, name=None)<br/>

[//]: # Given a list x and a list y, this operation returns a list out that represents all numbers that are in x but not in y.<br/>
[//]: # out: A Tensor. Has the same type as x. 1-D. Values present in x but not in y.<br/>
[//]: # idx: A Tensor of type int32. 1-D. Positions of x values preserved in out.<br/>
tf.listdiff(x, y, name=None)<br/>
[//]: # Returns: A tuple of Tensor objects (y, idx). y: A Tensor. Has the same type as x. 1-D. idx: A Tensor of type int32. 1-D.<br/>
tf.unique(x, y,name=None)<br/>

[//]: # y[x[i]] = i for i in [0, 1, ..., len(x) -1]<br/>
tf.invert_permutation(x, name=None)<br/>

## session
tf.Session.__init__(target='', graph=None, config=None)

[//]: # feches 就是需要实现的方法， feed_dict是填入的数据<br/>
[//]: # if fetches是方法， 返回null， 如果是 tensor 则返回ndarray<br/>
tf.Session.run(fetches, feed_dict=None)

[//]: # 与common session 没多大区别<br/>
tf.InteractiveSession()

## value
tf.zeros(shape, dtype, name) -> tensor
tf.ones(shape, dtype,name) -> tensor

[//]: # 与tensor一样的shape 的全为零的tensor<br/>
tf.zeros_like(tensor, dtype=None, name=None) -> tensor
tf.ones_like(tensor, dtype=None, name=None) -> tensor

[//]: # dims就是shape, value 为int32或,float32<br/>
tf.fill(dims, value, name=None) ->tensor

tf.constant(value, dtype=None, shape=None, name='Const') ->tensor<br/>
tf.range(start, limit, delta=1, name='range')<br/>
tf.linspace(start, stop, num, name=None)<br/>

[//]: # Randomly shuffles a tensor along its first dimension.<br/>
tf.random_shuffle(value, seed=None, name=None)<br/>
tf.set_random_seed(seed)

## Control Flow Operations:
[//]: # 返回一个一模一样的节点<br/>
tf.identity(input, name=None) -> tensor

[//]: # 并行计算tensors<br/>
tf.tuple(tensors, name=None, control_inputs=None)

tf.logical_not(x, name=None)<br/>
tf.logical_and(x, y, name=None)<br/>
tf.logical_or(x, y, name=None)<br/>
tf.logical_xor(x, y, name='LogicalXor')<br/>
tf.equal(x, y, name=None)<br/>
tf.not_equal(x, y, name=None)<br/>
tf.less(x, y, name=None)<br/>
tf.less_equal(x, y, name=None)<br/>
tf.greater(x, y, name=None)<br/>
tf.greater_equal(x, y, name=None)<br/>

[//]: # output should be taken from t (if true) or e (if false).<br/>
[//]: # condition、t、e the same shape, condition is fill of true or false<br/>
tf.select(condition, t, e, name=None)                               
[//]: # 不会

[//]: # This operation returns the coordinates of true elements in input.<br/>
[//]: # 就是 True 的位置<br/>
tf.where(input, name=None)

## framework

tf.Graph.__init__()

[//]: # 2. Constructing and making default:<br/>
[//]: # with tf.Graph().as_default() as g:<br/>
[//]: #   c = tf.constant(5.0)<br/>
tf.Graph.as_default()<br/>

[//]: # Finalizes this graph, making it read-only. use for ensure  thread safe<br/>
tf.Graph.finalize()

[//]: # with g.control_dependencies([a, b]):<br/>
  [//]: # Ops declared here run after `a` and `b`.<br/>
[//]: #   with g.control_dependencies([c, d]):<br/>
    [//]: # Ops declared here run after `a`, `b`, `c`, and `d`.<br/>
tf.Graph.control_dependencies(control_inputs)


[//]: # with g.name_scope("inner"):<br/>
[//]: #      nested_inner_1_c = tf.constant(30.0, name="c")<br/>
[//]: #      assert nested_inner_1_c.name == "nested/inner_1/c"<br/>
tf.Graph.name_scope(name)

tf.Graph.add_to_collection(name, value)<br/>
tf.Graph.get_collection(name, scope=None)<br/>
tf.Graph.get_operation_by_name(name)<br/>
tf.Graph.get_tensor_by_name(name)<br/>
tf.Graph.get_operations()<br/>

[//]: # a shortcut for calling tf.get_default_session().run(t).<br/>
tf.Tensor.eval(feed_dict=None, session=None)

[//]: # get_shape().as_list()<br/>
tf.Tensor.get_shape() ->TensorShape<br/>
tf.Tensor.set_shape(shape)<br/>
tf.TensorShape.as_list()<br/>

tf.get_default_graph()<br/>

tf.placeholder(dtype, shape=None, name=None) -> tensor<br/>

## Readers
class tf.ReaderBase<br/>
class tf.TextLineReader<br/>
class tf.WholeFileReader<br/>
class tf.IdentityReader<br/>
class tf.TFRecordReader<br/>
class tf.FixedLengthRecordReader<br/>



## Activation Functions
tf.nn.relu(features, name=None)<br/>
tf.nn.relu6(features, name=None)<br/>
tf.nn.softplus(features, name=None)<br/>
tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)<br/>
tf.nn.bias_add(value, bias, name=None)<br/>
tf.sigmoid(x, name=None)<br/>
tf.tanh(x, name=None)<br/>

## convolution

tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)<br/>
[//]: # 深度卷积<br/>
tf.nn.depthwise_conv2d(input, filter, strides, padding, name=None)                            [//]: # 不会<br/>
[//]: # 深度分离卷积<br/>
tf.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, name=None) [//]: # 不会<br/>

## pooling

tf.nn.max_pool(value, ksize, strides, padding, name=None)<br/>
tf.nn.avg_pool(value, ksize, strides, padding, name=None)<br/>
[//]: # return output and index of Targmax The flattened indices of the max values chosen for each output.<br/>
tf.nn.max_pool_with_argmax(input, ksize, strides, padding, Targmax=None, name=None)<br/>

## Normalization

[//]: # output = x / sqrt(max(sum(x**2), epsilon))<br/>
tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)<br/>

[//]: # LRN 局部响应归一化<br/>
tf.nn.local_response_normalization(input, depth_radius=None, bias=None, alpha=None, beta=None, name=None)[//]: # 不会<br/>


[//]: # return  Two Tensors: mean and variance.<br/>
tf.nn.moments(x, axes, name=None)<br/>


## losses
[//]: # output = sum(t ** 2) / 2 <br/>
tf.nn.l2_loss(t, name=None)

## Classification
[//]: # 二分类<br/>
[//]: # 它适用于每个类别相互独立但互不排斥的情况,在一张图片中，同时包含多个分类目标（大象和狗），那么就可以使用这个函数。<br/>
[//]: # max(x, 0) - x * z + log(1 + exp(-abs(x)))<br/>
tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None)<br/>

[//]: # Computes softmax activations.<br/>
tf.nn.softmax(logits, name=None)<br/>

[//]: # 它适用于每个类别相互独立且排斥的情况，一幅图只能属于一类，而不能同时包含一条狗和一只大象<br/>
tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)<br/>
[//]: # 如果labels的每一行是one-hot表示，也就是只有一个地方为1，其他地方为0，可以使用tf.sparse_softmax_cross_entropy_with_logits()<br/>
[//]: # sparse 与非sparse的区别: sparse label输入的是int型 softmax的label 输入的是one-hot<br/>
tf.nn.sparse_softmax_cross_entropy_with_logits(sentinel=None,labels=None,logits=None, name=None)<br/>

[//]: # SparseTensor(values=[1, 2], indices=[[0, 0], [1, 2]], shape=[3, 4])<br/>
[//]: #   [[1, 0, 0, 0]<br/>
[//]: #    [0, 0, 2, 0]<br/>
[//]: #    [0, 0, 0, 0]]<br/>
tf.SparseTensor.__init__(indices, values, shape)<br/>

[//]: # 用 default_value 填充 sp_input<br/>
tf.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value, name=None)<br/>
tf.sparse_tensor_to_dense(sp_input, default_value, name=None)<br/>

[//]: # output[d_0, d_1, ..., d_n, sp_input[d_0, d_1, ..., d_n, k]] = True
tf.sparse_to_indicator(sp_input, vocab_size, name=None)<br/>


tf.sparse_concat(concat_dim, sp_inputs, name=None)<br/>
[//]: # 对内部的排序进行reorder,实际位置并没有变化<br/>
tf.sparse_reorder(sp_input, name=None)<br/>

[//]: # 对内部元素进行删除, to_retain = [True, False,False, True]<br/>
tf.sparse_retain(sp_input, to_retain)<br/>

[//]: # 对没有值的行 进行 填充， specified default_value at index [row, 0]<br/>
tf.sparse_fill_empty_rows(sp_input, default_value, name=None)<br/>


## Variables
tf.Variable.__init__(initial_value, trainable=True, collections=None, validate_shape=True, name=None)<br/>
tf.Variable.initialized_value()<br/>
tf.get_variable(name, shape=None, dtype=tf.float32, initializer=None, trainable=True, collections=None)<br/>
tf.get_variable_scope()<br/>
tf.variable_scope(name_or_scope, reuse=None, initializer=None)<br/>
tf.constant_initializer(value=0.0)<br/>


tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)<br/>
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)<br/>
tf.random_uniform(shape, minval=0.0, maxval=1.0, dtype=tf.float32, seed=None, name=None)<br/>
tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None)<br/>
tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None)<br/>
tf.random_uniform_initializer(minval=0.0, maxval=1.0, seed=None)<br/>
tf.uniform_unit_scaling_initializer(factor=1.0, seed=None)<br/>
tf.zeros_initializer(shape, dtype=tf.float32)<br/>

[//]: # use_locking: If True, use locking during the assignment.<br/>
tf.Variable.assign(value, use_locking=False)<br/>
tf.Variable.assign_add(delta, use_locking=False)<br/>
tf.Variable.assign_sub(delta, use_locking=False)<br/>
tf.Variable.scatter_sub(sparse_delta, use_locking=False)<br/>
tf.Variable.eval(session=None)<br/>
tf.Variable.count_up_to(limit)<br/>
tf.Variable.get_shape()<br/>

[//]: # Returns all variables collected in the graph.<br/>
tf.all_variables()<br/>
[//]: # Returns all variables created with trainable=True.<br/>
tf.trainable_variables()<br/>
[//]: # This is just a shortcut for initialize_variables(all_variables())<br/>
tf.initialize_all_variables()<br/>
[//]: # Returns an Op to check if variables are initialized.<br/>
tf.assert_variables_initialized(var_list=None)<br/>

## Saver

[//]: # var_list can be a list of variables or a dict of names to variables:<br/>
[//]: # reshape: If True, allows restoring parameters from a checkpoint where the variables have a different shape.<br/>
[//]: # sharded: If True, shard the checkpoints, one per device.<br/>
[//]: # max_to_keep: maximum number of recent checkpoints to keep. Defaults 5.<br/>
[//]: # keep_checkpoint_every_n_hours: How often to keep checkpoints. Defaults to 10,000 hours.<br/>

tf.train.Saver.__init__(var_list=None,<br/>
                    reshape=False,<br/>
                    sharded=False,<br/>
                    max_to_keep=5,<br/>
                    keep_checkpoint_every_n_hours=10000.0,<br/>
                    name=None, restore_sequentially=False, saver_def=None,<br/> builder=None)

tf.train.Saver.save(sess, save_path, global_step=None, latest_filename=None)<br/>
tf.train.Saver.restore(sess, save_path)<br/>
tf.train.Saver.last_checkpoints<br/>
tf.train.Saver.set_last_checkpoints(last_checkpoints)<br/>
tf.train.update_checkpoint_state(save_dir, model_checkpoint_path, all_model_checkpoint_paths=None, latest_filename=None)<br/>


tf.train.Optimizer.__init__(use_locking, name)<br/>
tf.train.Optimizer.minimize(loss, global_step=None, var_list=None, gate_gradients=1, name=None)<br/>
tf.train.Optimizer.compute_gradients(loss, var_list=None, gate_gradients=1)<br/>
tf.train.Optimizer.apply_gradients(grads_and_vars, global_step=None, name=None)<br/>


tf.train.GradientDescentOptimizer.__init__(learning_rate, use_locking=False, name='GradientDescent')<br/>
tf.train.AdagradOptimizer.__init__(learning_rate, initial_accumulator_value=0.1, use_locking=False, name='Adagrad')<br/>
tf.train.MomentumOptimizer.__init__(learning_rate, momentum, use_locking=False, name='Momentum')<br/>
tf.train.AdamOptimizer.__init__(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')<br/>
tf.train.FtrlOptimizer.__init__(learning_rate, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='Ftrl')<br/>
tf.train.RMSPropOptimizer.__init__(learning_rate, decay, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp')<br/>
tf.stop_gradient(input, name=None)<br/>


tf.global_norm(t_list, name=None)<br/>
tf.train.Optimizer.get_slot_names()<br/>
tf.train.Optimizer.get_slot(var, name)<br/>

## Decaying the learning rate

[//]: # global_step = tf.Variable(0, trainable=False)<br/>
[//]: # starter_learning_rate = 0.1<br/>
[//]: # learning_rate = tf.exponential_decay(starter_learning_rate, global_step,100000, 0.96, staircase=True)<br/>
[//]: # optimizer = tf.GradientDescent(learning_rate)<br/>
[//]: # Passing global_step to minimize() will increment it at each step.<br/>
[//]: # optimizer.minimize(...my loss..., global_step=global_step)<br/>
tf.train.exponential_decay(learning_rate, global_step, decay_steps,<br/> decay_rate, staircase=False, name=None)<br/>

class tf.train.ExponentialMovingAverage<br/>
