import seaborn as sb
import argParser as parse
import time
import sys
import psutil
import logging
# from models import g2k_lstm_mcr as mcr
from torch.utils.data import DataLoader
import networkx_graph as nx_g
import load_traj as load
import tensorflow as tf
import json
import re
from tqdm import tqdm
import helper
# from torch.utils.data import dataloader
import numpy as np
import torch
import torch.nn.functional as F
import relational_inf_models.nri_learned as nri
from matplotlib import pyplot as py
# import tensorflow.python.util.deprecation as deprecation
import os
import glob
from strggrnn_model import strggrnn
import tensorboardX as TX
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

# print(tf.executing_eagerly())
# print(tf.config.experimental_functions_run_eagerly())


def train(args):

    # tf.config.run_functions_eagerly(True)
    out_graph = tf.Graph()
    # tf.compat.v1.enable_v2_behavior()
    # tf.compat.v1.enable_eager_execution()
    # with out_graph.as_default():
    #     tf.compat.v1.enable_eager_execution()
    #     tf.config.run_functions_eagerly(True)
    #     print(tf.version.VERSION)
    # print(tf.executing_eagerly())
    # print(tf.config.experimental_functions_run_eagerly())
    STRGGRNN_model_train(out_graph, args, infer=False)

    # out_graph.close()

def main():
    # sys.argv = ['-f']
    args = parse.ArgsParser()
    train(args.parser.parse_args())
    return

# TODO use glstm to predict path along with neighborhood boundaries using inner estimated soft attention mechanism.
# take relative distance + orthogonality between people vislets (biased by visual span width)
# extract outputs from glstm as follows: neighborhood influence (make message passing between related pedestrians)
# then transform using mlp the new hidden states mixture into (x,y) euclidean locations.


def evaluate(params, n_proposals, i):
    ade_tmp, final_tmp, pred_tmp = [], [], []

    def condition(i, num_nodes_sc, target_traj, pred_path,
                      euc_loss, fde, num_end_targets):
        return tf.less(i, num_nodes_sc)

    @tf.function
    def inner_loop_fn(i, num_nodes_sc, target_traj, pred_path,
                      euc_loss, fde, num_end_targets):
        try:
            # Calculating error when predicting the future offset in (x,y) coordinates
            euc_loss = tf.subtract(target_traj, tf.add(pred_path, target_traj))
            fde = tf.subtract(target_traj[:, -1, :], tf.add(pred_path[:, -1, :], target_traj[:, -1, :]))

        except KeyError:
            print('Key Error inside loop')
            pass

        i = tf.add(i, 1)
        num_end_targets = tf.add(num_end_targets, 1)

        return i, num_nodes_sc, \
               target_traj, pred_path, euc_loss, fde, num_end_targets

    num_nodes_sc, target_traj, pred_path, \
    euc_loss, fde, num_end_targets = params

    for k in range(n_proposals):
        p = pred_path[k][0:len(target_traj)] # pred_path[k][0:target_traj.shape[0]]
        pred_path_var = tf.convert_to_tensor(value=p)

        loop_vars = [i, num_nodes_sc,
                     target_traj, pred_path_var, euc_loss, fde,
                     num_end_targets]

        i, num_nodes_sc, \
        target_traj, _, euc_loss, fde, num_end_targets = tf.while_loop(cond=condition, body=inner_loop_fn, loop_vars=loop_vars,
                                                                       parallel_iterations=n_proposals)

        ade_tmp.append(euc_loss)
        final_tmp.append(fde)
    return tf.stack(ade_tmp), tf.stack(final_tmp)


@tf.function
def assess_rcmndr(pred_len, num_nodes, pred_path, target_traj0_ten, model,  n_proposals=10):
    tf.config.run_functions_eagerly(True)  # run tf function decorator and its operations without session
    pred_path = tf.transpose(a=pred_path, perm=(0, 3, 2, 1))
    i = tf.zeros(shape=(), dtype=tf.float32)
    num_nodes0 = tf.convert_to_tensor(num_nodes, dtype=tf.float32)
    num_end_targets = tf.zeros(shape=())
    euc_loss = tf.zeros((num_nodes, pred_len, 2), dtype=np.float32)
    fde = tf.zeros((num_nodes, 2), dtype=np.float32)

    params = [num_nodes0, target_traj0_ten, pred_path, euc_loss, fde, num_end_targets]

    ade_losses, fde_loss = evaluate(params, n_proposals, i)
    # ade_losses = tmp/pred_len # / num_nodes  # (len(euc_loss) * batch_len)
    ade_stack_2d = tf.linalg.norm(ade_losses, axis=3)
    ade_stack = tf.reduce_mean(ade_stack_2d, axis=2) #[:int(pred_len/2)]
    ade_stack = tf.reduce_mean(ade_stack, axis=1)
    # ade_stack = tf.reduce_mean(ade_losses, axis=2)
    # ade_stack = tf.linalg.norm(ade_stack, axis=2)
    # ade_stack = tf.reduce_mean(ade_stack, axis=1)

    # print(tf.compat.v2.executing_eagerly())

    model.ade_op.assign(ade_stack, use_locking=True, read_value=False, name='ass_ade_op')
    ade_min_idx = tf.Variable(tf.argmin(model.ade_op.read_value()))
    ade_loss_final = tf.reduce_min(model.ade_op.read_value())
    fde_loss = tf.reduce_min(tf.reduce_mean(tf.linalg.norm(fde_loss, axis=2), axis=1)) #/ num_nodes

    # fde_loss = tf.reduce_sum(tf.linalg.norm(fde_loss, axis=2), axis=1) / num_nodes
    # t1 = tf.reduce_sum(ade_losses, axis=2)/12
    # t2 = tf.linalg.norm(t1, axis=2)
    # t3 = tf.reduce_sum(t2, axis=1) / (num_nodes)
    # ade_loss_final = tf.reduce_min(t3)

    # TODO pick minimum then optimize
    return ade_loss_final, fde_loss, ade_min_idx.read_value()


def STRGGRNN_model_train(out_graph, args, infer=0):
    # with tf.compat.v1.Session(graph=out_graph).as_default() as out_sess:
    # with tf.compat.v1.Session(graph=out_graph).as_default() as sess:
    tf.config.run_functions_eagerly(True)
    parent_dir = '/home/sisi/Documents/Sirius_challenge/'
    summary_logdir = parent_dir + 'traj_rcmndr-master/TF/data/train/TBX'
    tboard_summary = TX.SummaryWriter(logdir=summary_logdir)

    log_count_f = open(parent_dir + 'log/strggrnn_counts.txt', 'w')
    log_dir = open(parent_dir + 'log/strggrnn_ade_log_kfold.csv', 'w')
    log_dir_fde = open(parent_dir + 'log/strggrnn_fde_log_kfold.csv', 'w')
    save_dir = parent_dir + 'save/ggrnnv/strggrnn_{}'.format(args.leaveDataset)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    pid = psutil.Process(os.getpid())

    # time_log = open(os.path.join(parent_dir, train_loader.used_data_dir, 'training_Tlog.txt'), 'w')

    graph = nx_g.online_graph(args)
    num_nodes = 0
    pred_path = []
    e = 0
    frame = 0
    num_targets = 0
    num_end_targets = 0
    batch = {}
    # K-Fold cross validation scheme
# if not infer:
    # train_data = TrajectoryDataset(parent_dir + 'Sirius_json', 'train') # 'traj_rcmndr-master/TF/data/', 'train'
    # train_loader = DataLoader(
    #     dataset=train_data,
    #     batch_size=32, #args.batch_size,
    #     num_workers=2,
    #     collate_fn=torch_collate_fn,
    #     shuffle=True,
    # )
    # print(train_loader.sel_file)
    train_loader = load.DataLoader(args=args, path=parent_dir + 'traj_rcmndr-master/TF/data/',
                                   leave=args.leaveDataset, infer=infer)
    train_loader.reset_data_pointer()
    flag = True
    # opt_op = loss_optzr.minimize(loss=loss, var_list=l2norm_vec, name='loss_optzr')
    best_val = 100
    while e < args.num_epochs:
        e_start = time.time()
        print('Epoch Started = ', e)
        print('                  ############################################                       ')
        # train_loader.rl_num_nodes = train_data.total
        # train_loader = train_loader
        gt_traj, future_traj, _ = train_loader.read_batch()
        batch["coords"] = list(gt_traj.values())
        batch["future_coords"] = list(future_traj.values())

        # for batch_idx, btraj in enumerate(train_loader):

        try:
            # enum_traj = enumerate(train_loader)
            # for batch_idx, batch in enum_traj:
            for batch_idx in range(train_loader.num_batches):
                # tf.compat.v1.initialize_all_variables().run(session=sess)
                # loss_optzr = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
                # loss_optzr.minimize(loss=model.l2loss, var_list=model.l2norm_vec, name='loss_optzr')
                print('#################  Batch Started = ', batch_idx+1, ' of {} #################'.format(train_loader.num_batches))
                start_all = time.time()
                rcmndr_start = time.time()
                cust_dataloader = {'train_loaderobj': train_loader, 'batch': batch}
                # train_loader.rl_num_nodes = len(batch['coords'])
                # model = strggrnn(inputs=[args, out_graph, graph, out_sess, data_loader, train_loader.frame_pointer])
                model = strggrnn(inputs=[args, out_graph, graph, cust_dataloader, (batch_idx * args.pred_len), False])
                if e == 0 and batch_idx == 0:
                    loss_optzr = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
                    @tf.function
                    def _l2loss():
                        return tf.nn.l2_loss(t=model.l2norm_vec, name='loss')

                    # checkpoint_path = os.path.join(save_dir, 'strggrnn_train_{0}.ckpt'.format(e))
                    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=loss_optzr, model=model, root=model)
                    ckpt_saver = tf.train.CheckpointManager(ckpt, save_dir, checkpoint_name="ggrnn_v_train", max_to_keep=None)

                model.fit()
                # tf.compat.v1.get_default_graph().get_collection(name='variables', scope='krnl_weights').clear()
                # train_loader.tick_frame_pointer(hop=args.gt_frame_hop)
                eager_assess = tf.function(assess_rcmndr)
                tf.config.run_functions_eagerly(True)
                # print(tf.compat.v2.executing_eagerly())
                ade_bst, fde_bst, min_idx = eager_assess(args.pred_len,
                                                         model.rl_num_nodes, model.all_pred_path,
                                                         model.nri_obj.target_traj0_ten,
                                                         model=model, n_proposals=model.nri_obj.n_proposals)
                np_ade_bst = ade_bst.numpy()
                np_fde_bst = fde_bst.numpy()

                # model.bst_adj_prop = model.nri_obj.adj_mat_vec[min_idx]
                # model.krnl_mdl.hidden_states = tf.matmul(model.bst_adj_prop, model.hidden_state)
                # model.krnl_mdl.hidden_states = tf.nn.softmax(model.krnl_mdl.hidden_states)
                # preds = torch.tensor(model.all_pred_path[0].numpy())
                # preds = preds.permute(2, 1, 0)
                # t_gt_future = torch.tensor(model.nri_obj.target_traj0_ten.numpy())
                # preds = torch.add(preds[:len(t_gt_future)], t_gt_future)
                # num_of_agents = model.rl_num_nodes
                # total_ade[0] += num_of_agents * torch.mean(
                #     F.pairwise_distance(preds[:, :20].contiguous().view(-1, 2),
                #                         batch['future_coords'][:, :20].contiguous().view(-1, 2))).item()
                # total_ade[1] += num_of_agents * torch.mean(
                #     F.pairwise_distance(preds[:, :40:2].contiguous().view(-1, 2),
                #                         batch['future_coords'][:, :40:2].contiguous().view(-1, 2))).item()
                # total_ade[2] += num_of_agents * torch.mean(
                #     F.pairwise_distance(preds[:, :80:4].contiguous().view(-1, 2),
                #                         batch['future_coords'][:, :80:4].contiguous().view(-1, 2))).item()
                #
                # total_fde[0] += torch.sum(F.pairwise_distance(preds[:, 19].reshape(-1, 2),
                #                                               batch['future_coords'][:, 19].reshape(-1, 2))).item()
                #
                # num_nodes += num_of_agents
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
                print('ADE = ', np_ade_bst)
                print('FDE = ', np_fde_bst)

                # nri_obj.l2norm_vec = tf.Variable(tf.convert_to_tensor(euc_min))
                # TODO: think about optimizing at each prediction or optimizing after batch
                start = time.time()
                model.l2norm_vec.assign(ade_bst, use_locking=True, read_value=False, name='ass_l2norm')
                # tf.Tensor(op=assign_op, value_index=0, dtype=tf.float32)
                # l2loss = lambda : tf.nn.l2_loss(t=model.l2norm_vec.read_value(), name='loss')
                loss_optzr.minimize(loss=_l2loss, var_list=[model.l2norm_vec],
                                    name='optzr_min_op')
                loss = model.l2norm_vec.read_value().numpy()

                print('Loss = ', loss)
                tboard_summary.add_scalar(tag='ADE', scalar_value=np_ade_bst, global_step=e*batch_idx)
                tboard_summary.add_scalar(tag='Loss', scalar_value=loss, global_step=e*batch_idx)
                # loss_optzr.run()
                # sess.run(loss_optzr)#, feed_dict={l2norm_vec: euc_disp})
                print('BackPropagation with SGD took:{}'.format(time.time()-start))
                # bst_adj_prop = model.nri_obj.adj_mat_vec[min_idx.eval()]
                # model.krnl_mdl.hidden_states = tf.matmul(bst_adj_prop, model.hidden_state)
                # model.krnl_mdl.hidden_states = tf.nn.softmax(model.krnl_mdl.hidden_states)
                # make it stateful when batch_size is small enough to find trajectories
                # related to each other between 2 consecutive batches .
                # model.hidden_state = model.krnl_mdl.hidden_states
                num_targets += model.rl_num_nodes

                print('\nFull pipeline time =', time.time() - start_all)
                # time_log.write('{0},{1},{2}\n'.format(e, time.time() - model.start_b,
                #                (pid.memory_info().rss / 1024 / 1024 / 1024)))

                # log_dir.write('{0},{1}\n'.format(e, np_ade_bst))
                # log_dir_fde.write('{0},{1}\n'.format(e, np_fde_bst))

                tboard_summary.add_scalar(tag='RAM used', scalar_value=pid.memory_info().rss / 1024 / 1024 / 1024)
                tboard_summary.flush()

                print('============================')
                print("Memory used: {:.2f} GB".format(pid.memory_info().rss / 1024 / 1024 / 1024))
                print('============================')

                end_t = time.time()
                logger.warning('{0} seconds to complete'.format(end_t - model.start_t))
                # logger.warning('Batch = {3} of {4} Frame {0} Loss = {1}, num_ped={2}'
                #                .format(train_loader.frame_pointer, loss, model.rl_num_nodes, batch_idx,
                #                        args.batch_size)) #len(train_loader)

                if (e * batch_idx) and (e * batch_idx) % args.save_every == 0 : #e % args.save_every == 0 and and (e * batch_idx) == 5
                    # batch.pop("coords")
                    # batch.pop("future_coords")
                    # del batch
                    print('Saving model at epoch {0}'.format(e))
                    ckpt.step.assign_add(1)
                    ckpt_saver.save(checkpoint_number=(e * batch_idx))
                    print("model saved to {}".format(save_dir))
        except TypeError:
            pass
        e += 1
        e_end = time.time()

        print('Epoch time taken: {}'.format(e_end - e_start))
        log_count_f.write('ADE steps {0}\nFDE steps = {1}'.format(num_targets, num_end_targets))
        ckpt_saver = tf.train.CheckpointManager(ckpt, save_dir, checkpoint_name="ggrnn_best_val", max_to_keep=None)
        # val_loader = load.DataLoader(args=args, path=parent_dir + 'traj_rcmndr-master/TF/data/',
        #                                leave=args.leaveDataset)
        # Validation step
        # val_data = TrajectoryDataset(parent_dir + 'Sirius_json', 'val')
        # val_loader = DataLoader(
        #     val_data,
        #     batch_size=64,
        #     num_workers=2,
        #     collate_fn=torch_collate_fn,
        #     shuffle=False,
        # )

        # Check for invalid Json data:
        try:
            val_loader = load.DataLoader(args=args, path=parent_dir + 'traj_rcmndr-master/TF/data/',
                                           leave=args.leaveDataset, infer=True)
            # val_loader.reset_data_pointer(dataset_pointer=args.leaveDataset, frame_pointer=0)
            gt_traj, future_traj, _ = val_loader.read_batch()
            batch["coords"] = list(gt_traj.values())
            batch["future_coords"] = list(future_traj.values())
            # enum_traj = enumerate(val_loader)
            # for batch_idx, batch in enum_traj:
            for batch_idx in range(val_loader.num_batches):
                cust_dataloader = {'train_loaderobj': val_loader, 'batch': batch}

                # data_loader = {'train_loaderobj': cust_dataloader, 'batch': batch}
                # train_loader.rl_num_nodes = len(batch['coords'])
                # model = strggrnn(inputs=[args, out_graph, graph, out_sess, data_loader, train_loader.frame_pointer])
                model = strggrnn(inputs=[args, out_graph, graph, cust_dataloader, (batch_idx * args.pred_len), False])
                model.fit()

                eager_assess = tf.function(assess_rcmndr)
                tf.config.run_functions_eagerly(True)
                ade_bst, fde_bst, min_idx = eager_assess(args.pred_len,
                                                         model.rl_num_nodes, model.all_pred_path,
                                                         model.nri_obj.target_traj0_ten,
                                                         model=model, n_proposals=model.nri_obj.n_proposals)
                np_ade_bst = ade_bst.numpy()
                np_fde_bst = fde_bst.numpy()
                print('Val ADE = ', np_ade_bst)
                print('Val FDE = ', np_fde_bst)
                # preds = torch.tensor(model.all_pred_path[min_idx].numpy())
                # preds = preds.permute(2, 1, 0)
                # t_gt_future = torch.tensor(model.nri_obj.target_traj0_ten.numpy())
                # preds = torch.add(preds[:len(t_gt_future)], t_gt_future)
                num_of_agents = model.rl_num_nodes

                total_ade[0] += num_of_agents * np_ade_bst

                total_fde[0] += num_of_agents * np_fde_bst
                num_nodes += num_of_agents
            print('total Val ADE = ', total_ade[0] / num_nodes)
            print('total Val FDE = ', total_fde[0] / num_nodes)
            if best_val > total_ade[0] / num_nodes:
                best_val = total_ade[0] / num_nodes
                ckpt.step.assign_add(1)
                ckpt_saver.checkpoint.save(checkpoint_number=(e * batch_idx))
        except TypeError:
            pass

    # log_f.close()
    # sess.close()
    # time_log.close()
    log_count_f.close()
    log_dir.close()
    log_dir_fde.close()


if __name__ == '__main__':
    main()
