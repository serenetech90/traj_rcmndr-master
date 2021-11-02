import tensorflow as tf


class gsk_lstm_cell(tf.Module):
    def __init__(self, in_features, obs_len, num_nodes, lambda_reg, def_g):
        super().__init__()
        self.out_size = tf.compat.v1.placeholder_with_default(input=num_nodes, shape=[], name='out_size')
        self.lambda_reg = tf.Variable(lambda_reg, dtype=tf.float32)
        self.init_w = tf.compat.v1.initializers.random_normal(mean=0, stddev=0.5, seed=0, dtype=tf.float32)
        self.def_g = def_g

        with tf.compat.v1.variable_scope(name_or_scope="krnl_vars", reuse=True):
            self.embedded_spatial_vislet = tf.compat.v1.placeholder(dtype=tf.float32)

            self.outputs = tf.compat.v1.placeholder_with_default(input=tf.random.normal(shape=[int(in_features.shape[0]),
                                                 int(in_features.shape[0])], mean=0, stddev=1, seed=0, dtype=tf.float32),
                                                 shape=[int(in_features.shape[0]), int(in_features.shape[0])], name="outputs")

            self.ngh = tf.compat.v1.placeholder_with_default(input=tf.random.normal(shape=[int(in_features.shape[0]), 8],
                                                                                    mean=0, stddev=0.05, seed=0, dtype=tf.float32),
                                                                                    shape=[int(in_features.shape[0]), 8], name="ngh")

            self.pred_path_band = tf.Variable(initial_value=tf.random.normal(shape=[10, 2, 12, num_nodes],
                                                                        mean=0, stddev=1, seed=0,
                                                                        dtype=tf.float32), shape=[10, 2, 12, num_nodes],
                                                                        name="pred_path_band")

        with tf.compat.v1.variable_scope(name_or_scope="krnl_weights", reuse=True):
            self.weight_v = tf.compat.v1.placeholder_with_default(name='weight_v', input= \
                                        self.init_w(shape=(8, int(in_features.shape[0]))),
                                        shape=(8, int(in_features.shape[0])))
                                        # dtype=tf.float32)

            self.bias_v = tf.compat.v1.placeholder_with_default(name='bias_v', input= \
                                      self.init_w(shape=(int(in_features.shape[0]),)),
                                      shape=(int(in_features.shape[0])),)
                                      # dtype=tf.float32)

            self.weight_o = tf.compat.v1.placeholder_with_default(name='weight_o', input= \
                                        self.init_w(shape=(int(in_features.shape[0]), num_nodes)),
                                        shape=(int(in_features.shape[0]), num_nodes))
                                        # dtype=tf.float32)

            self.weight_c = tf.compat.v1.placeholder_with_default(name='weight_c', input= \
                                        self.init_w(shape=(24, obs_len)),
                                        shape=(24, obs_len))
                                        # dtype=tf.float32)

    @tf.function
    def run(self, num_nodes, k):
        # with self.def_g.as_default():
        self.embedded_spatial_vislet = tf.matmul(self.weight_v, self.outputs) + self.bias_v  # 12x10
        # ngh_temp = tf.Variable((self.lambda_reg * self.ngh) )# 12x10
        # self.outputs = tf.Variable(self.outputs)
        # self.ngh = tf.Variable(tf.multiply(self.ngh, tf.transpose(self.embedded_spatial_vislet)))
        self.ngh = tf.multiply(self.ngh, tf.transpose(self.embedded_spatial_vislet))
        _, self.cost = tf.gradients(ys=self.ngh, xs=[self.embedded_spatial_vislet, self.ngh],
                                    stop_gradients=self.embedded_spatial_vislet,
                                    unconnected_gradients='zero')

        self.cost = tf.nn.relu(
            tf.squeeze(tf.matmul(tf.matmul(self.weight_v, self.outputs) + self.bias_v, self.ngh)))
        # self.cost = tf.Variable(self.cost)
        # 24x10
        self.temp_path = (tf.matmul(tf.matmul(self.weight_c, tf.transpose(self.cost)), self.weight_o))  # 24xn

        self.pred_path_band[k].assign(tf.reshape(self.temp_path, (2, 12, num_nodes)))  # 2x12xn

            # return self.pred_path_band


    # self.visual_path = tf.placeholder_with_default(input=tf.random.normal(shape=[1, int(in_features.shape[0])],
    #                                                                       mean=0, stddev=1, seed=0,
    #                                                                       dtype=tf.float32),  # dtype=tf.float32,
    #                                                shape=[1, in_features.shape[0]], name="visual_path")


