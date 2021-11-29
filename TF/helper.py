# from torch.utils.serialization import load_lua as llua
import tensorflow as tf
from tensorflow.python.ops import rnn_cell as rnn
# https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/rnn/python/ops/rnn_cell.py
# implementation of GNN (Gated Neighborhood Network)
# we make use of peephole connections as grid lstm emerged from highway networks

class neighborhood_stat_vis_loc_encoder():
    #  TODO code
    #     define correlation between vislets and locations and let the higher correlations define
    #     the zone of interest (region of interest that is influential to pedestrian, encode temporal features
    #     about the active hotspot/ salient interaction spot)
    #     try rnn as encoder for features
    #     for now multiple cues are concatenated.
    #  TODO NEW *************************************************************************************
    #   Compare 2 parallel GridLSTM memory and time with 3 parallel LSTM streams for vis, loc, stat_ngh

    def __init__(self, hidden_size,hidden_len, num_layers, grid_size, embedding_size, dim, obs_len, dropout=0):
        # super(neighborhood_vis_loc_encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        # self.inputs = tf.placeholder(dtype=tf.float64, shape=[obs_len, obs_len], name="inputs")

        # TODO: shape 4D
        # self.state_f00_b00_c = tf.compat.v1.placeholder(name='state_f00_b00_c', shape=(obs_len, self.hidden_size), dtype=tf.float64)
        # self.c_hidden_state = tf.compat.v1.placeholder(name='c_hidden_state', shape=(hidden_len, self.hidden_size), dtype=tf.float64)

        # self.state_f00_b00_m = tf.compat.v1.placeholder(name='state_f00_b00_m', shape=(hidden_len, (grid_size * int(grid_size/2))), dtype=tf.float64)
        # self.num_freq_blocks = tf.compat.v1.placeholder(name='num_freq_blocks', dtype=tf.float32)
        self.hidden_size = hidden_size
        # self.stat_input = tf.compat.v1.placeholder(name='stat_input', shape=(dim+2, obs_len), dtype=tf.float64)

        # self.hidden_state = tf.compat.v1.placeholder(name='hidden_state', shape=(dim+2, self.hidden_size), dtype=tf.float64)

        # self.output = tf.compat.v1.placeholder(dtype=tf.float64, shape=[hidden_len, hidden_len], name="output")

        self.rnn = rnn.GridLSTMCell(num_units=num_layers,
                                    feature_size=grid_size,
                                    frequency_skip=grid_size,
                                    use_peepholes=True,
                                    num_frequency_blocks=[int(obs_len/grid_size)], #int(grid_size/2)
                                    share_time_frequency_weights=True,
                                    state_is_tuple=False,
                                    couple_input_forget_gates=True,
                                    reuse=tf.compat.v1.AUTO_REUSE)

        self.stat_rnn = rnn.GridLSTMCell(num_units=num_layers,
                                        feature_size=grid_size,
                                        frequency_skip=grid_size,
                                        use_peepholes=False,
                                        num_frequency_blocks=[int(obs_len / grid_size)],
                                        share_time_frequency_weights=True,
                                        state_is_tuple=False,
                                        couple_input_forget_gates=True,
                                        reuse=tf.compat.v1.AUTO_REUSE)

        # self.forward()

    def update_input_size(self, new_size):
        self.input = tf.compat.v1.placeholder(dtype=tf.float64, shape=[new_size, new_size], name="inputs")
        self.hidden_state = tf.compat.v1.placeholder(name='hidden_state', shape=(new_size, self.hidden_size), dtype=tf.float64)

    def __call__(self, *args, **kwargs):
        # Combine both features
        # hidden state should be [2 x num_nodes x rnn_size x rnn_size]
        inputs, state_f00_b00_c, stat_input, stat_hidden_state = kwargs.values()
        (self.stat_output, self.stat_c_hidden_states) = self.stat_rnn(stat_input, stat_hidden_state)
        (self.output, self.c_hidden_state) = self.rnn(inputs=tf.transpose(inputs), state=state_f00_b00_c)

        return self.stat_output, self.stat_c_hidden_states, self.output, self.c_hidden_state

    def init_hidden(self, size):
        return tf.zeros(name='hidden_state',shape=(size, self.hidden_size), dtype=tf.float64)



#
# class neighborhood_stat_enc():
#     # TODO code for later experiment
#     #   define correlation static neighborhood and pedestrians relative distances to context
#     #   run kernel over region around pedestrian to specify static context features and determine
#     #   the static spot that influence pedestrian path
#     #   we need context features (cnn ?)
#     #   or grid of fixed lstm placements for top view.
#     #   lstm grid network can be customized for on-board camera.
#     #   consider the frame discretization in (Choi et al 2019)
#     #   they used convolutional features for H-S and H-H
#     #    install a grid checkpoints once, pedestrian enter a checkpoint share parameters from previous checkpoint
#     #    between new checkpoints and its surrounding checkpoints
#     #    each checkpoint forms potential neighborhood and it's a small region of the static scene
#     #  how is this different from social lstm 2D grid mask?
#     #  how is this different from sharing pattern in LSTMGrid?
#     #  In order for us to know how neighborhood forms we need to
#     #  understand pedestrians awareness state and know how they perceive
#     #  environmental and social factors
#     #  passing features between past neighborhood and new neighborhoods and
#     #  their surrounding neighborhoods will enable the model of anticipating future area of anvigation and strict a band of future trajectory
#     #  the idea here is to use grid ecnoder comprised of LSTM instead of cnn (convolutions)
#     #  the passed and combined (summation) spatio-temporal features will be passed through soft-attention for weighing out the salient ROI
#     #  and the random walker will take the pedestrian spatial features for predicting relations and completing the edges
#     # Social-LSTM 2D grid is occupancy filter for pooling inside single neighborhood area
#     # test stvp that pools future paths together and not only the present paths
#     # after sharing parameters with new checkpoint run random walker to anticipate future region of interest that pedestrian is likely to navigate through
#     # based on their social interactions and vfoa
#     # our grid LSTM is responsible for controlling message passing about pedestrians H-S and H-H states between neighborhoods
#
#     # TODO MX-LSTM deploys on Basic LSTM cell implementation of Social LSTM
#     # try grid lstm cell in place of cnn to encode spatial features
#     # to encode relative spatial interactions human-space combine both neighborhood networks,
#     # the static neighborhood network encodes the poi in the scene first
#     def __init__(self, hidden_size, num_layers,grid_size, dim):
#         # super(neighborhood_stat_enc, self).__init__()
#
#         self.hidden_size = hidden_size
#
#         self.input = tf.compat.v1.placeholder(name='input', shape=(dim, 8),
#                                      dtype=tf.float64)
#
#         self.hidden_state = tf.compat.v1.placeholder(name='hidden_state', shape=(dim, self.hidden_size),
#                                     dtype=tf.float64)
#
#         # ctxt_img = tf.convert_to_tensor(imread(ctxt_path[0]), dtype=tf.float64)
#         # ctxt_img = tf.convert_to_tensor(tf.pad(ctxt_img, paddings=tf.constant([[1, 1, ], [0, 1], [0, 0]])),
#         #                                 dtype=tf.float64)
#         #
#         # ctxt_img = tf.expand_dims(ctxt_img, axis=0)
#         # self._2dconv = tf.nn.conv2d(input=ctxt_img, filter=tf.random_normal(shape=[561, 711, 3, 1],dtype=tf.float64),
#         #                             padding='VALID', strides=[1, 1, 1, 1])
#         # self._2dconv = tf.squeeze(self._2dconv)
#         # in gridLSTM frequency blocks are the units of lstm stacking vertically
#         # while time LSTM spans horizontally
#         self.rnn = rnn.GridLSTMCell(num_units=num_layers,
#                                     feature_size=grid_size,
#                                     frequency_skip=grid_size,
#                                     use_peepholes=False,
#                                     num_frequency_blocks=[int(grid_size/2)],
#                                     share_time_frequency_weights=True,
#                                     state_is_tuple=False,
#                                     couple_input_forget_gates=True,
#                                     reuse=True)
#
#         self.output, self.c_hidden_states = self.rnn(self.input, self.hidden_state)
#
#         # self.forward(static_mask , social_frame, hidden_state)
#         # self.rnn = nn.GRUCell(input_size, hidden_size, num_layers)
#         # GRUCell is stacked GRU model but it doesnt provide Grid scheme of sharing weights along multidimensional GRU
#         # llua('/home/serene/PycharmProjects/multimodaltraj/grid-lstm-master/model/GridLSTM.lua')
#
#     # def forward(self, input, hidden):
#         # map locations []
#         # use social frame in filling occupancy state of the static mask.
#         # opposed to occupancy map in Social LSTM (2016) which used binary occupancy descriptors
#         # our occupancy map has weighted descriptors.
#         # In algebra multiplying two feature vector calculates distance
#         # between two corresponding points in Euclidean geometry
#         # such that this relative distance embeddings now leads us to weight the frame neighborhoods and
#         # evaluate occupancy inside each local neighborhood
#         # input = tf.matmul(input, tf.ones_like(tf.transpose(input)))
#         # input = tf.transpose(input)
#
#
#         # return output, new_hidden
#
