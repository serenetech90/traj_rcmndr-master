import seaborn as sb
import argParser as parse
import time
import sys
import psutil
import logging
# from models import g2k_lstm_mcr as mcr

import networkx_graph as nx_g
import load_traj as load
import tensorflow as tf
import helper
from torch.utils.data import dataloader
import numpy as np
import torch
import torch.nn.functional as F
import relational_inf_models.nri_learned as nri
from matplotlib import pyplot as py
# import tensorflow.python.util.deprecation as deprecation
import os
import glob
from strggrnn_model import strggrnn
from iterator import TrajectoryDataset, torch_collate_fn
# import skimage.measure as sk

# reduce tf messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# deprecation._PRINT_DEPRECATION_WARNINGS = False

# Public vars
true_path = []
total_ade = [0,0,0]
total_fde = [0,0,0]

target_traj = []
pred_path = []
e = 0
frame = 1
num_targets = 0
num_end_targets = 0
attn = []

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.WARNING, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def loss(l2norm_vec):
    return tf.nn.l2_loss(t=l2norm_vec, name='loss')


# TODO use glstm to predict path along with neighborhood boundaries using inner estimated soft attention mechanism.
# take relative distance + orthogonality between people vislets (biased by visual span width)
# extract outputs from glstm as follows: neighborhood influence (make message passing between related pedestrians)
# then transform using mlp the new hidden states mixture into (x,y) euclidean locations.


def evaluate(params, n_proposals, i, i0):
    ade_tmp, final_tmp, pred_tmp = [], [], []

    def condition(pred_len, num_nodes_sc, num_nodes,
                  target_traj, pred_path, euc_loss, fde, num_end_targets, euc_min, euc_idx, i, i0):
        return tf.less(i, num_nodes_sc)

    def inner_loop_fn(pred_len, num_nodes_sc, num_nodes, target_traj, pred_path,
                      euc_loss, fde, num_end_targets, euc_min, euc_idx, i, i0):
        try:
            euc_loss = tf.subtract(target_traj, tf.add(pred_path, target_traj))
            # euc_loss = rel_future_path - target_traj
            fde = tf.subtract(target_traj[:, -1, :], tf.add(pred_path[:, -1, :], target_traj[:, -1, :]))

        except KeyError:
            print('Key Error inside loop')
            pass
        i = tf.add(i, 1)
        num_end_targets = tf.add(num_end_targets, 1)

        return pred_len, num_nodes_sc, num_nodes, \
               target_traj, pred_path, euc_loss, fde, num_end_targets, \
               euc_min, euc_idx, i, i0

    pred_len, num_nodes_sc, num_nodes, target_traj, pred_path, \
    euc_loss, fde, num_end_targets, euc_min, euc_idx = params

    # print('target_traj = ', target_traj)
    # print('pred_path = ', pred_path)
    for k in range(n_proposals):
        p = pred_path[k][0:target_traj.shape[0].value]
        # target_traj0_var = tf.convert_to_tensor(value=target_traj)
        pred_path_var = tf.convert_to_tensor(value=p)

        loop_vars = [pred_len, num_nodes_sc,
                     num_nodes, target_traj, pred_path_var, euc_loss, fde,
                     num_end_targets, euc_min, euc_idx, i, i0]
        # shapes = [pred_len.get_shape(), num_nodes_sc.get_shape(),
        #           tf.TensorShape([]),
        #           tf.TensorShape([None, 20, 2]), tf.TensorShape([None, 20, 2]),
        #           euc_loss.get_shape(), fde.get_shape(), num_end_targets.get_shape(),
        #           euc_min.get_shape(), euc_idx.get_shape(), i.get_shape(), i0.get_shape()]

        pred_len, num_nodes_sc, num_nodes, \
        target_traj, _, euc_loss, fde, num_end_targets, \
        euc_min, euc_idx, i, i0 = \
            tf.while_loop(cond=condition, body=inner_loop_fn, loop_vars=loop_vars, parallel_iterations=n_proposals)

        ade_tmp.append(euc_loss)
        final_tmp.append(fde)
    return tf.stack(ade_tmp), tf.stack(final_tmp)


def assess_rcmndr(sess, graph_t, pred_len, num_nodes, batch_len, euc_loss, fde, pred_path,
                  target_traj0_ten, model, attn=None, hidden_state=None, n_proposals=10):

    pred_path = tf.transpose(a=pred_path, perm=(0, 3, 2, 1))

    i0 = tf.zeros(shape=())
    pred_len0 = tf.convert_to_tensor(value=pred_len)
    i = tf.zeros(shape=(), dtype=tf.float32)
    num_nodes0 = tf.convert_to_tensor(num_nodes, dtype=tf.float32)
    # hidden_state0 = tf.convert_to_tensor(hidden_state)
    num_end_targets = tf.zeros(shape=())
    euc_min = tf.convert_to_tensor(np.inf, dtype=tf.float32)
    euc_idx = tf.zeros(shape=(1), dtype=tf.float32)

    euc_loss = np.zeros((num_nodes, pred_len, 2), dtype=np.float32)
    fde = np.zeros((num_nodes, 2), dtype=np.float32)

    euc_loss0 = tf.convert_to_tensor(euc_loss)
    fde0 = tf.convert_to_tensor(fde)

    params = [pred_len0, num_nodes0, num_nodes, target_traj0_ten, #tf.convert_to_tensor(target_traj0_ten), tf.convert_to_tensor(pred_path)
              pred_path, euc_loss0,  # euc_loss0, fde0
              fde0, num_end_targets, euc_min, euc_idx]

    ade_losses, fde_loss = evaluate(params, n_proposals, i, i0)

    ade_stack = tf.linalg.norm(ade_losses, axis=3)
    # ade_losses = tmp/pred_len # / num_nodes  # (len(euc_loss) * batch_len)
    model.ade_stack = tf.reduce_mean(tf.reduce_mean(ade_stack, axis=2), axis=1)
    ade_min_idx = tf.argmin(model.ade_stack)
    ade_loss_final = tf.reduce_min(model.ade_stack)
    fde_loss = tf.reduce_min(tf.reduce_mean(tf.linalg.norm(fde_loss, axis=2), axis=1)) #/ num_nodes

    # TODO pick minimum then optimize
    return ade_loss_final, fde_loss, ade_min_idx


def l2loss(err):
    return tf.nn.l2_loss(t=err, name='loss')


def STRGGRNN_model_train(out_graph, args):
    with tf.compat.v1.Session(graph=out_graph).as_default() as out_sess:
        parent_dir = '/home/sisi/Documents/Sirius_challenge/'
        train_loader = load.DataLoader(args=args, path=parent_dir + 'traj_rcmndr-master/TF/data/', leave=args.leaveDataset)
        time_log = open(os.path.join(parent_dir, train_loader.used_data_dir, 'training_Tlog.txt'), 'w')
        log_count_f = open(parent_dir + 'log/strggrnn_counts.txt', 'w')
        log_dir = open(parent_dir + 'log/strggrnn_ade_log_kfold.csv', 'w')
        log_dir_fde = open(parent_dir + 'log/strggrnn_fde_log_kfold.csv', 'w')
        save_dir = parent_dir+'save/'

        # TODO implement k-fold cross validation + check why pred_path is all zeros (bug in GridLSTMCell)
        graph = nx_g.online_graph(args)

        # print(train_loader.sel_file)
        train_loader.reset_data_pointer()
        flag = True
        pid = psutil.Process(os.getpid())
        num_nodes = 0
        with tf.compat.v1.Session(graph=out_graph) as sess:
            pred_path = []
            e = 0
            frame = 0
            num_targets = 0
            num_end_targets = 0
            batch = {}

            # opt_op = loss_optzr.minimize(loss=loss, var_list=l2norm_vec, name='loss_optzr')

            while e < args.num_epochs:
                e_start = time.time()
                train_data = TrajectoryDataset(parent_dir + 'traj_rcmndr-master/TF/data/', 'train') #'Sirius_json', 'train')
                train_loader = train_loader(
                    train_data,
                    batch_size=args.batch_size,
                    num_workers=2,
                    collate_fn=torch_collate_fn,
                    shuffle=True,
                )
                # train_loader.rl_num_nodes = train_data.total
                train_loader = train_loader
                gt_traj, future_traj, _ = train_loader.read_batch()
                batch["coords"] = list(gt_traj.values())
                batch["future_coords"] = list(future_traj.values())

                print('session started at Epoch = ', e)

                # for batch_idx, btraj in enumerate(train_loader):
                for batch_idx in range(args.batch_size):
                    start_all = time.time()

                    rcmndr_start = time.time()
                    data_loader = {'train_loaderobj': train_loader, 'batch': batch}

                    # model = strggrnn(inputs=[args, out_graph, graph, out_sess, data_loader, train_loader.frame_pointer])
                    model = strggrnn(inputs=[args, out_graph, graph, out_sess, data_loader, (batch_idx * args.pred_len)])
                    model.fit()
                    train_loader.tick_frame_pointer(hop=args.gt_frame_hop)
                    ade_bst, fde_bst, min_idx = assess_rcmndr(sess, out_graph, args.pred_len,
                                                              train_loader.rl_num_nodes, model.rl_num_nodes, total_ade,
                                                              total_fde, model.all_pred_path, model.nri_obj.target_traj0_ten, model,
                                                              model.nri_obj.n_proposals)

                    if e == 0:
                        tf.compat.v1.initialize_all_variables().run(session=sess)
                        loss_optzr = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
                        # loss_optzr.minimize(loss=model.l2loss, var_list=model.l2norm_vec, name='loss_optzr')

                    preds = torch.tensor(model.all_pred_path[0])
                    preds = preds.permute(2, 1, 0)
                    preds = torch.add(preds, torch.tensor(model.nri_obj.target_traj0_ten.eval()))
                    num_of_agents = model.rl_num_nodes
                    # total_ade[0] += num_of_agents * torch.mean(
                    #     F.pairwise_distance(preds[:, :20].contiguous().view(-1, 2),
                    #                         batch['future_coords'][:, :20].contiguous().view(-1, 2))).item()
                    # total_ade[1] += num_of_agents * torch.mean(
                    #     F.pairwise_distance(preds[:, :40:2].contiguous().view(-1, 2),
                    #                         batch['future_coords'][:, :40:2].contiguous().view(-1, 2))).item()
                    # total_ade[2] += num_of_agents * torch.mean(
                    #     F.pairwise_distance(preds[:, :80:4].contiguous().view(-1, 2),
                    #                         batch['future_coords'][:, :80:4].contiguous().view(-1, 2))).item()

                    # total_fde[0] += torch.sum(F.pairwise_distance(preds[:, 19].reshape(-1, 2),
                    #                                               batch['future_coords'][:, 19].reshape(-1, 2))).item()

                    num_nodes += num_of_agents
                    # total_fde[1] += torch.sum(F.pairwise_distance(preds[:, 39].reshape(-1, 2),
                    #                                               batch['future_coords'][:, 39].reshape(-1, 2))).item()
                    # total_fde[2] += torch.sum(F.pairwise_distance(preds[:, 79].reshape(-1, 2),
                    #                                               batch['future_coords'][:, 79].reshape(-1, 2))).item()

                    rcmndr_end = time.time()
                    # print('\n time taken to recommend best adjacency proposal: {}\n'.format(rcmndr_end - rcmndr_start))

                    # ade_losses = euc_min / (model.nri_obj.n_proposals * args.pred_len)
                    # fde_min = tf.reduce_min(fde_min)
                    # min_idx = tf.argmin(ade_losses)
                    #
                    # ade_min = tf.reduce_min(ade_losses)

                    # bst_adj_prop = model.nri_obj.adj_mat_vec[min_idx.eval()]
                    # print('Short ADE', total_ade[0] / num_nodes)
                    # print('Short ADE', total_fde[0] / num_nodes)
                    print('ADE = ', ade_bst.eval())
                    print('FDE = ', fde_bst.eval())

                    # nri_obj.l2norm_vec = tf.Variable(tf.convert_to_tensor(euc_min))
                    # TODO: think about optimizing at each prediction or optimizing after batch
                    start = time.time()
                    model.l2norm_vec.assign(ade_bst, use_locking=True, read_value=False).run()
                    loss_optzr.minimize(loss=model.l2norm_vec, name='loss_optzr').run()
                    # loss_optzr.run()
                    # sess.run(loss_optzr)#, feed_dict={l2norm_vec: euc_disp})
                    print('BackPropagation with SGD took:{}'.format(time.time()-start))

                    # model.krnl_mdl.hidden_states = tf.matmul(bst_adj_prop, model.hidden_state)
                    # model.krnl_mdl.hidden_states = tf.nn.softmax(model.krnl_mdl.hidden_states)
                    # make it stateful when batch_size is small enough to find trajectories
                    # related to each other between 2 consecutive batches .
                    # model.hidden_state = model.krnl_mdl.hidden_states
                    num_targets += model.rl_num_nodes

                    print('\nFull pipeline time =', time.time() - start_all)
                    time_log.write('{0},{1},{2}\n'.format(e, time.time() - model.start_b,
                                   (pid.memory_info().rss / 1024 / 1024 / 1024)))

                    log_dir.write('{0},{1}\n'.format(e, ade_bst))
                    log_dir_fde.write('{0},{1}\n'.format(e, fde_bst))

                    print('============================')
                    print("Memory used: {:.2f} GB".format(pid.memory_info().rss / 1024 / 1024 / 1024))
                    print('============================')

                    end_t = time.time()
                    logger.warning('{0} seconds to complete'.format(end_t - model.start_t))
                    logger.warning('Batch = {3} of {4} Frame {0} Loss = {1}, num_ped={2}'
                                   .format(train_loader.frame_pointer, model.l2loss().eval(), model.rl_num_nodes, batch_idx,
                                           args.batch_size)) #len(train_loader)

                if (e * batch_idx) % args.save_every == 0: #e % args.save_every == 0 and
                    print('Saving model at epoch {0}'.format( e))
                    checkpoint_path = os.path.join(save_dir, 'strggrnn_train_{0}.ckpt'.format(e))
                    saver = tf.compat.v1.train.Saver(out_graph.get_collection('trainable_variables')) #tf.compat.v1.all_variables())
                    saver.save(sess, checkpoint_path, global_step=e * batch_idx)

                    print("model saved to {}".format(checkpoint_path))

                e += 1
                e_end = time.time()

                print('Epoch time taken: {}'.format(e_end - e_start))
                log_count_f.write('ADE steps {0}\nFDE steps = {1}'.format(num_targets, num_end_targets))
        # log_f.close()
        sess.close()
        time_log.close()
        log_count_f.close()
        log_dir.close()
        log_dir_fde.close()
    out_sess.close()


def train(args):
    out_graph = tf.Graph()
    # tf.enable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    # tf.compat.v1.enable_eager_execution()
    STRGGRNN_model_train(out_graph, args)


def main():
    # sys.argv = ['-f']
    args = parse.ArgsParser()
    train(args.parser.parse_args())
    return


if __name__ == '__main__':
    main()
