import time
import numpy as np
from models import g2k_lstm_mcr as mcr
# from models import gsk_lstm_cell as cell
from multiprocessing import Pool
import threading
import tensorflow as tf
import helper
import relational_inf_models.nri_learned as nri


class strggrnn(tf.Module):
    def __init__(self, inputs):
        super().__init__()
        args, tf_graph, graph, dataloader, frame, infer = inputs
        self.args = args
        self.onlinegraph = graph
        self.out_graph = tf_graph
        # self.sess = sess
        self.frame = frame
        self.pool = Pool()

        # self.dataloader = dataloader['dataloaderobj']
        self.dataloader = dataloader['train_loaderobj']
        self.batch_coords = dataloader['batch']["coords"]
        self.vislet = list(dataloader['train_loaderobj'].traj_vislet.values())

        # if infer:
        #     self.batch_future_coords = []
        # else:
        self.batch_future_coords = dataloader['batch']["future_coords"]

        # self.rl_num_nodes = self.dataloader.rl_num_nodes
        self.nri_obj = nri.nri_learned(args=args)
        self.dim = int(args.neighborhood_size / args.grid_size)
        # self.pred_path = tf.compat.v1.placeholder(dtype=tf.float32)

        self.init_w = tf.initializers.RandomNormal(mean=0, stddev=1, seed=0)

        self.loss = tf.Variable(initial_value=self.init_w(shape=()), trainable=True, dtype=tf.float32)
        self.hidden_filters = tf.keras.initializers.truncated_normal()(shape=[8, 1, 1], dtype=tf.float32)
        self.krnl_mdl = None
        self.batch_v = None
        self.rl_num_nodes = 0

        self.nghood_enc = helper.neighborhood_stat_vis_loc_encoder(
            hidden_size=args.rnn_size,
            hidden_len=self.dim,
            num_layers=args.num_layers,
            grid_size=args.grid_size,
            embedding_size=args.embedding_size,
            dim=self.dim,
            obs_len=args.obs_len,
            dropout=args.dropout)
        # self.nri_obj = nri.nri_learned(args=args, sess=None)

        init = tf.initializers.RandomNormal(mean=0, stddev=1, seed=0)
        self.l2norm_vec = tf.Variable(initial_value=init(()), dtype=tf.float32, shape=(), name='l2norm_vec')
        self.ade_op = tf.Variable(initial_value=init(shape=(10,), dtype=tf.float32), shape=(10,), name='ade_op')

        stat_mask = tf.zeros(shape=(self.dim, args.obs_len), dtype=tf.float32)
        stat_mask += tf.expand_dims(
            tf.range(start=0, limit=1, delta=(1 / args.obs_len), dtype=tf.float32), axis=0)
        self._2dconv_in = stat_mask #self.dataloader._2dconv_in

        #tf.zeros(shape=(self.dim + 2, args.obs_len), dtype=tf.float32)

        self.stat_hidden_state = tf.zeros(dtype=tf.float32, shape=(self._2dconv_in.shape[0], args.rnn_size))

    @tf.function
    def l2loss(self):
        return tf.reduce_sum(tf.pow(self.l2norm_vec, 2)) / 2

    def fit(self, infer=False):
        # only init hidden states at 1st epoch, 1st batch, no resetting is happening after that

        graph_t = self.onlinegraph.ConstructGraph(current_batch=self.batch_coords, vislets=self.vislet, framenum=0,
                                                  future_traj=self.batch_future_coords)
        # self.rl_num_nodes = self.dataloader.rl_num_nodes = self.batch.shape[0] #sirius dataset

        # TODO: validate whether networkx class restores the node values correctly.
        self.batch_v = list(graph_t.get_node_attr(param='node_pos_list').values()) #self.batch['coords']
        self.batch_v = tf.linalg.norm(tf.stack(self.batch_v), axis=2)
        self.batch_v = tf.transpose(self.batch_v)
        self.rl_num_nodes = self.dataloader.rl_num_nodes = self.batch_v.shape[1] #len(self.batch_coords)
        self.hidden_state = tf.keras.initializers.random_normal()(shape=(self.args.obs_len, self.args.rnn_size)) #self.rl_num_nodes

        self.vislet = tf.stack(list(graph_t.get_node_attr(param='vislets').values()))
        self.vislet = tf.linalg.norm(tf.stack(self.vislet), axis=2)
        self.vislet = tf.transpose(self.vislet)
        # self.dataloader.min_max_coords = [np.min(np.stack(self.batch)[self.frame:][:][0]),
        #                                   np.min(np.stack(self.batch)[:][:][1]),
        #                                   np.max(np.stack(self.batch)[:][:][0]),
        #                                   np.max(np.stack(self.batch)[:][:][1])]
        # if len(np.array(self.batch).shape) > 2:
            # if self.frame > 0:
        # self.batch = self.batch[:][self.frame:self.frame + self.args.obs_len]
            # else:
            #     self.batch = self.batch[:][self.frame:self.frame + self.args.obs_len]

        # else:
        #     # self.batch = np.array(self.batch)[frame:frame + args.obs_len]
        #     # self.dataloader.reset_data_pointer()
        #     pass

        # self.nri_obj.target_traj0_ten = self.batch_future_coords
        self.batch_future_coords = list(graph_t.get_node_attr(param='targets').values())
        self.nri_obj.target_traj0_ten = tf.stack(self.batch_future_coords)

        # self.nri_obj.target_traj0_ten = self.sess.run([self.nri_obj.target_traj0_ten], {
        #         self.nri_obj.target_traj0_ten: nri.extract_ten_dict(graph_t.get_node_attr('targets'))}
        # )

        # self.sess.run([self.pred_path], {
        # self.pred_path: tf.keras.initializers.random_normal()(shape=(10, 2, self.args.pred_len, self.rl_num_nodes), dtype=tf.float32).eval()})[0]

        # self.krnl_mdl = MC.g2k_lstm_mc(in_features=self.nghood_enc.input,  # MC.g2k_lstm_mc
        #                                num_nodes=self.rl_num_nodes,
        #                                obs_len=self.args.obs_len,
        #                                lambda_reg=self.args.lambda_param,
        #                                sess_g=self.out_graph)

        # with tf.variable_creator_scope('weight_input', reuse=True):
        weight_i = tf.Variable(name='weight_i',
                               initial_value=self.init_w(shape=(self.args.obs_len, self.rl_num_nodes)),
                               trainable=True, dtype=tf.float32)
        weight_ii = tf.Variable(name='weight_ii',
                                initial_value=self.init_w(shape=(self.rl_num_nodes, self.args.obs_len)),
                                trainable=True, dtype=tf.float32)
        weight_vi = tf.Variable(name='weight_vi',
                                initial_value=self.init_w(
                                    shape=(self.vislet.shape[1], 2)),
                                trainable=True, dtype=tf.float32)

        # tf.compat.v1.initialize_variables(var_list=[weight_i, weight_ii]).run()

        # Embed features set into fixed-shaped compact tensor [8x8]
        # tf.compat.v1.initialize_variables(var_list=[self.krnl_mdl.weight_c, self.krnl_mdl.weight_o,self.krnl_mdl.bias_v,
        #                                             self.krnl_mdl.weight_v]).run()
        # tf.compat.v1.initialize_variables(var_list=[self.krnl_mdl.attn]).run()
        # tf.compat.v1.initialize_variables(var_list=[self.krnl_mdl.ngh, self.krnl_mdl.outputs]).run()
        # tf.compat.v1.initialize_variables([self.krnl_mdl.cost]).run()

        # try:
        #     self.sess.run(fetches=tf.compat.v1.initialize_all_variables())
        # except tf.errors.FailedPreconditionError:
        #     self.sess.run(fetches=tf.compat.v1.initialize_all_variables())
        # inputs = tf.convert_to_tensor(self.batch_v, dtype=tf.float32)
        # inputs = tf.matmul(inputs, weight_i)

        # inputs_emb = tf.multiply(self.batch_v, weight_i)
        inputs_emb = tf.multiply(weight_i, self.batch_v)
        inputs_emb = tf.matmul(inputs_emb, weight_ii)
        vislet_emb = tf.matmul(self.vislet, weight_vi)
        # vislet_emb = tf.matmul(tf.transpose(self.vislet), weight_vi)
        vislet_past = vislet_emb
        vislet_rel = vislet_past * vislet_emb
        inputs_emb = tf.concat((inputs_emb, tf.transpose(vislet_emb)), axis=0)
        # vislet_rel = tf.matmul(tf.transpose(vislet_emb), weight_ii)
        # inputs = tf.concat((inputs_emb, tf.transpose(vislet_emb)), axis=0)
        # tf.multiply(inputs, vislet_emb)  #

        self.start_t = time.time()

        # compute relative locations and relative vislets
        f_hat, new_hidden_state, ng_output, c_hidden_state = self.nghood_enc(inputs=inputs_emb,
                                                                             state_f00_b00_c=self.hidden_state,
                                                                             # output=self.out_init,
                                                                             stat_input=self._2dconv_in,
                                                                             c_hidden_state=self.stat_hidden_state)

        # ng_output = tf.matmul(tf.transpose(ng_output), weight_ii)
        self.krnl_mdl = mcr.g2k_lstm_mcr(in_features=f_hat,  # MC.g2k_lstm_mc
                                         # vislets=tf.transpose(vislet_emb),
                                         num_nodes=self.rl_num_nodes,
                                         obs_len=self.args.obs_len,
                                         pred_len=self.args.pred_len,
                                         hidden_size=self.args.rnn_size,
                                         lambda_reg=self.args.lambda_param)

        # self.krnl_mdl = cell.gsk_lstm_cell(in_features=f_hat,
        #                                    num_nodes=self.rl_num_nodes,
        #                                    obs_len=self.args.obs_len,
        #                                    pred_len=self.args.pred_len,
        #                                    lambda_reg=self.args.lambda_param,
        #                                    def_g=self.out_graph)
        if infer:
            for k, v in self.bst_krnl_mdl_vals.items():
                if k in ['lambda_reg', 'weight_c', 'weight_v', 'weight_r', 'bias_v']:
                    self.krnl_mdl.__setattr__(k, v)

        # f_hat, new_hidden_state, ng_output, c_hidden_state = \
            # self.sess.run([self.nghood_enc.input, self.nghood_enc.state_f00_b00_c,
            #                self.nghood_enc.output, self.nghood_enc.c_hidden_state],
            #               feed_dict={self.nghood_enc.input: inputs.eval(),
            #                          self.nghood_enc.state_f00_b00_c: self.hidden_state.eval(),
            #                          self.nghood_enc.output: self.out_init.eval(),
            #                          self.nghood_enc.stat_input: self._2dconv_in.eval(),
            #                          self.nghood_enc.c_hidden_state: self.c_hidden_init.eval()})

        # GGRNN-V , STR-GGRNN-V
        # tr_hidden_state = tf.reshape(self.hidden_state, shape=(self.args.obs_len, 128, 1))
        # _conv1d = tf.nn.conv1d(input=tr_hidden_state,
        #                        filters=self.hidden_filters,
        #                        padding='VALID', stride=[1, 6, 1])

        tr_hidden_state = self._2dconv_in #tf.concat((self._2dconv_in, vislet_emb), axis=1)
        # fully-connected dense multiplication to transform to [20x20]
        # tr_hidden_state = tf.matmul(tf.transpose(tr_hidden_state), tr_hidden_state)
        # tr_hidden_state = tf.abs(tr_hidden_state)
        # tr_hidden_state = tf.nn.softmax(tf.exp(tr_hidden_state) / tf.cumsum(tf.exp(tr_hidden_state)))
        # self.krnl_attn = self.krnl_mdl.attn.eval()
        t_attn_W = tf.nn.softmax(tf.exp(self.krnl_mdl.attn) / tf.cumsum(tf.exp(self.krnl_mdl.attn)))

        start = time.time()
        self.nri_obj.adj_mat_vec = self.nri_obj.h_to_a(num_nodes_sc=self.args.maxNumPeds,
                                                       h_comp=tr_hidden_state.numpy(), w=t_attn_W.numpy())

        print('time taken to generate proposals:{}: '.format(time.time() - start))
        self.start_b = time.time()

        # def _inner(self, k):
        #     self.krnl_mdl.outputs = f_hat
        #     self.krnl_mdl.out_size = self.rl_num_nodes
        #     # self.krnl_mdl.pred_path_band = self.all_pred_path[k]
        #     self.krnl_mdl(self.rl_num_nodes, k)
            # pred_path_np, prob_mat = \
                # self.sess.run([self.krnl_mdl.pred_path_band, self.krnl_mdl.cost],
                #               feed_dict={
                #                   self.krnl_mdl.outputs: f_hat,
                #                   self.krnl_mdl.out_size: self.rl_num_nodes,
                #                   self.krnl_mdl.pred_path_band: self.all_pred_path[k]#.eval(session=self.sess)
                #               })
            # self.all_pred_path[k] = self.sess.run(self.krnl_mdl.pred_path_band)

        # self.all_pred_path = tf.convert_to_tensor(self.all_pred_path)
        # self.all_pred_path = tf.Variable(self.krnl_mdl.init_w(shape=(10, 2, self.args.pred_len, self.rl_num_nodes)))
        # threads = [] * self.nri_obj.n_proposals
        # processes = self.pool.map(func=rang, iterable=range(self.nri_obj.n_proposals))
        # for k, _ in enumerate(processes):
        # if not infer:
        for k, _ in enumerate(range(self.nri_obj.n_proposals)):
            # self.krnl_mdl.outputs = f_hat
            # self.krnl_mdl.out_size = self.rl_num_nodes
            self.krnl_mdl(k, ng_output, tf.concat((f_hat, vislet_emb), axis=1), vislet_rel, self.hidden_state)

            # tf.pad(nri_obj.adj_mat_vec[k], paddings=[[1,1], [1,1]])
            # print('k = ', k)
            # threads[k] = threading.Thread(target=_inner, args=(self, k))
            # threads[k].start()
            # threads[k].join()

        self.all_pred_path = self.krnl_mdl.pred_path_band.read_value()
        t_attn_W = self.krnl_mdl.attn.numpy()

        # self.nri_obj.adj_mat_vec = self.nri_obj.h_to_a(num_nodes_sc=self.rl_num_nodes,
        #                                                h_comp=np.random.normal(size=(self.rl_num_nodes, self.rl_num_nodes)),
        #                                                w=t_attn_W)

        # make it stateful when batch_size is small enough to find trajectories
        # related to each other between 2 consecutive batches .
        self.hidden_state = self.krnl_mdl.hidden_states
        # else:
        #     self.krnl_mdl(0)
        #     self.all_pred_path = self.krnl_mdl.pred_path_band.read_value()[0]

        print('\nTime taken evaluate 10 predictions per each pedestrian + Social recommendation = ',
              time.time() - self.start_b)


def rang(x):
    return x-1

