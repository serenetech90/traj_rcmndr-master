import tensorflow as tf

class gsk_lstm_cell(tf.Module):
    def __init__(self, in_features, obs_len, pred_len, num_nodes, lambda_reg, def_g):
        super().__init__()
        self.out_size = num_nodes #tf.Variable(initial_value=num_nodes, shape=[], name='out_size')
        self.lambda_reg = tf.Variable(initial_value=lambda_reg, dtype=tf.float32)
        self.init_w = tf.initializers.random_normal(mean=0, stddev=0.5, seed=0)
        self.pred_len = pred_len
        self.def_g = def_g

        with tf.compat.v1.variable_scope(name_or_scope="krnl_vars", reuse=True):
            # self.embedded_spatial_vislet = tf.compat.v1.placeholder(dtype=tf.float32)

            self.embedded_spatial_vislet = tf.Variable(initial_value=self.init_w(shape=[in_features.shape[0], obs_len]), dtype=tf.float32)
            self.outputs = tf.Variable(initial_value=in_features, dtype=tf.float32, name="outputs")

            self.ngh = tf.Variable(initial_value=self.init_w(shape=[in_features.shape[0], obs_len]), dtype=tf.float32, name="ngh")

            self.pred_path_band = tf.Variable(initial_value=tf.random.normal(shape=[10, 2, pred_len, num_nodes],
                                              mean=0, stddev=1, seed=0,
                                              dtype=tf.float32), shape=[10, 2, pred_len, num_nodes],
                                              name="pred_path_band")

        with tf.compat.v1.variable_scope(name_or_scope="krnl_weights", reuse=True):
            self.weight_v = tf.Variable(name='weight_v', initial_value= \
                                        self.init_w(shape=(obs_len, in_features.shape[1])),
                                        shape=(obs_len, in_features.shape[1]))
                                        # dtype=tf.float32)

            self.bias_v = tf.Variable(name='bias_v', initial_value= \
                                      self.init_w(shape=(in_features.shape[1],)),
                                      shape=(in_features.shape[1]),)
                                      # dtype=tf.float32)

            self.weight_o = tf.Variable(name='weight_o', initial_value= \
                                        self.init_w(shape=(in_features.shape[0], num_nodes)),
                                        shape=(in_features.shape[0], num_nodes))
                                        # dtype=tf.float32)

            self.weight_c = tf.Variable(name='weight_c', initial_value= \
                                        self.init_w(shape=(2*self.pred_len, obs_len)),
                                        shape=(2*self.pred_len, obs_len))
                                        # dtype=tf.float32)

    def __call__(self, *args, **kwargs): #(self, num_nodes, k):
        super.__call__()
        # with self.def_g.as_default():
        k = args

        @tf.function
        def _inner_call():

            self.embedded_spatial_vislet = tf.add(tf.matmul(self.outputs, self.weight_v ), self.bias_v)  # [10x8]
            # ngh_temp = tf.Variable((self.lambda_reg * self.ngh) )# 12x10
            # self.outputs = tf.Variable(self.outputs)
            # self.ngh = tf.Variable(tf.multiply(self.ngh, tf.transpose(self.embedded_spatial_vislet)))
            # self.ngh = tf.multiply(self.ngh, self.embedded_spatial_vislet) # [10x8]
            # _, self.cost = tf.gradients(ys=self.ngh, xs=[self.embedded_spatial_vislet, self.ngh],
            #                             stop_gradients=self.embedded_spatial_vislet,
            #                             unconnected_gradients='zero')

            self.cost = tf.nn.relu(tf.squeeze(tf.multiply(self.embedded_spatial_vislet, self.ngh)))
            # self.cost = tf.Variable(self.cost)
            # 24x10
            self.temp_path = (tf.matmul(tf.matmul(self.weight_c, tf.transpose(self.cost)), self.weight_o))  # 2xself.pred_len x n

            self.pred_path_band[k].assign(tf.reshape(self.temp_path, (2, self.pred_len, self.out_size)))  # 2x12xn

        # execute tf decorator function as inner function in the original caller function
        _inner_call()
    # self.visual_path = tf.placeholder_with_default(input=tf.random.normal(shape=[1, int(in_features.shape[0])],
    #                                                                       mean=0, stddev=1, seed=0,
    #                                                                       dtype=tf.float32),  # dtype=tf.float32,
    #                                                shape=[1, in_features.shape[0]], name="visual_path")



