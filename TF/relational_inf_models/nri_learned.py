import numpy as np
from tensorflow import nn
import tensorflow as tf
import sklearn.decomposition as sk_dec
# from tensorflow.contrib import learn as skf
# import pathos.multiprocessing as mp
# from threading import Thread

# @tf.contrib.eager.defun

# pool = mp.Pool(processes=mp.cpu_count())


def extract_ten_dict(fdict):
    # rows_lens = []
    # due to limitations in converting to ragged tensor from list of variable-length tensors,
    # there is no other way in TF except fixing the size.
    # problem in casting type list to Tensor
    # max_ten = tf.zeros(shape=(len(dict), 12, 2))
    ten=[]

    for itr in list(fdict.keys()):
        if len(fdict[itr][0]) < 20: # max_ten[itr].shape[0]:
            # if itr < ten.shape[0]:
            fdict[itr] = np.array(fdict[itr], dtype=float)
            ten.append(np.concatenate((fdict[itr], np.zeros(shape=(1, abs(fdict[itr].shape[0] - 20), 2))), axis=1))
        else:
            ten.append(fdict[itr][0])
        # max_ten[itr] = tf.assign(max_ten[itr], (max_ten[itr] + dict[itr]))
        # rows_lens.append(int(dict[itr].shape[0]))

    return ten

class nri_learned():
     n_proposals = 10
     def __init__(self, args, sess):
        super(nri_learned, self).__init__()
        self.args = args
        self.sess = sess

        self.target_traj0_ten = tf.compat.v1.placeholder(name='target_traj0_ten', dtype=tf.float32)
        # self.adj_mat_vec = tf.Variable(dtype=tf.float32, name='adj_mat_vec')
        # self.threaded_fn = Thread(target=self.h_to_a, daemon=True)
        # self.threaded_fn.start()

     def h_to_a(self, num_nodes_sc, h_comp, w):
        # TODO construct kernel from random walk theory
        # TODO random walk is fast but least accurate model among graph completion algos
        # TODO check the literature for online nmf (OMF)
        # transform graph to kernel to parameterize
        # fNRI factorization of edges using the softmax
        #  make it variational, make n_proposals projections to generate n_proposals different permutations of adjacency
        # adj will all be ones, assuming fully connected graph at the init
        # use nmf to sparsify the adj, by making more plausible connections and less density graph.
        # TODO: restore all the adj_mat versions and select the best adjacency matrix for this pedestrian
        #  based on the least errors generated then run optimizer upon that.
        # adj = np.ones(shape=(h_comp.shape[0], h_comp.shape[0]))
        adj = np.ones(shape=(num_nodes_sc, num_nodes_sc), dtype=np.float32)
        adj_mat_vec = np.zeros(shape=(self.n_proposals, adj.shape[0], adj.shape[1]), dtype=np.float32)
        h_comp = np.array(h_comp, dtype=np.float32)
        w = np.array(w, dtype=np.float32)
        
        # print('Adjacency matrix length:{}'.format(num_nodes_sc))
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        # try:
        # while not coord.should_stop():
        # sess.run([hidden_state,attn], options=tf.RunOptions(timeout_in_ms=300))
        # except Exception as e:
        #     coord.request_stop(e)
        # finally:
        #     coord.request_stop()
        #     coord.join(threads)
        # w = np.pad(w , [[1,1],[1,1]], mode='minimum')
        for k in range(self.n_proposals):
            # TODO see if map_fn applicable for online nmf
            w, h, n_iter = sk_dec.non_negative_factorization(X=adj, H=w, W=h_comp, init='custom',
                                                             n_components=adj.shape[0])
            adj_mat = np.matmul(w, h)
            adj_mat_vec[k] = adj_mat
        # edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
        # prob = my_softmax(logits, -1)
        # loss_kl = kl_categorical_uniform(prob, args.num_atoms, edge_types)
        return adj_mat_vec

    # Through gated neighborhood network (neighborhood encoders & random walker)

     # def select_best_rlns(self,n_adj, g):
     #
     #    #  map h onto adj_mat using h_to_a() function
     #    # make NMF on this adj_mat
     #    # infer relationships from the kernel (kernel output by random walker algorithm)
     #    # need to be created once at the init of master network.
     #    prob_mat = nn.sigmoid(n_adj)
     #    return prob_mat

     def eval_rln_ngh(self,adj_mat, combined_ngh):
        # This is the same mechanism used for choosing best ngh as (SGTV, 2019)
        # evaluate importance of relations to form the hybrid neighborhood(social(temporal) + static(spatial))
        # prob_mat = nn.Sigmoid(adj_mat)
        prob_mat = nn.softmax(adj_mat)

        return prob_mat

