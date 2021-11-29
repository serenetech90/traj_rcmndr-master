import io
import time

import matplotlib.pyplot
import numpy as np
import seaborn as sb
import psutil
import networkx_graph as nx_g
import load_traj as load
import tensorflow as tf
import json
import re
import torch
import torch.nn.functional as F
import os
from strggrnn_model import strggrnn
import tensorboardX as TX
import argParser as parse
from train import assess_rcmndr

# Public vars
true_path = []
total_ade = [0, 0, 0]
total_fde = [0, 0, 0]

target_traj = []
pred_path = []
e = 0
frame = 1
num_targets = 0
num_end_targets = 0
attn = []


def test(args):
    out_graph = tf.Graph()
    STRGGRNN_model_test(out_graph, args, infer=True)


def main():
    # sys.argv = ['-f']
    args = parse.ArgsParser()
    test(args.parser.parse_args())
    return

def STRGGRNN_model_test(out_graph, args, infer=0):
# with tf.compat.v1.Session(graph=out_graph).as_default() as out_sess:
# with tf.compat.v1.Session(graph=out_graph).as_default() as sess:

    parent_dir = '/home/sisi/Documents/Sirius_challenge/'
    summary_logdir = parent_dir + 'traj_rcmndr-master/TF/data/train/TBX/zara2_test/'
    # tboard_summary = TX.GlobalSummaryWriter(logdir=summary_logdir)

    file_writer = tf.summary.create_file_writer(logdir=summary_logdir)
    file_writer.init()
    log_count_f = open(parent_dir + 'log/strggrnn_counts.txt', 'w')
    log_dir = open(parent_dir + 'log/strggrnn_ade_log_kfold.csv', 'w')
    log_dir_fde = open(parent_dir + 'log/strggrnn_fde_log_kfold.csv', 'w')
    save_dir = parent_dir + 'save/ggrnnv/strggrnn_{}'.format(args.leaveDataset)

    pid = psutil.Process(os.getpid())

    # time_log = open(os.path.join(parent_dir, train_loader.used_data_dir, 'training_Tlog.txt'), 'w')

    graph = nx_g.online_graph(args)
    num_nodes = 0

    batch = {}
    # inference mode (testing)
    # with open(parent_dir + 'Sirius_json/test_agents.json') as f:
    #     test = json.load(f)
    #
    # results = {k: {} for k in test.keys()}

    # data = TrajectoryDataset(parent_dir+'Sirius_json/', 'test')
    #
    # test_data_loader = DataLoader(
    #     data,
    #     batch_size=64,
    #     num_workers=2,
    #     collate_fn=torch_collate_fn,
    #     shuffle=False,
    # )
    test_loader = load.DataLoader(args=args, path=parent_dir + 'traj_rcmndr-master/TF/data/',
                                  leave=args.leaveDataset, infer=infer)
    print('Predicting and preparation of the submission file')
    # ckpt_reader = tf.train.load_checkpoint(save_dir+'ggrnn_best_val-2405')
    ckpt_reader = tf.train.load_checkpoint(tf.train.latest_checkpoint(save_dir))
    bst_model = ckpt_reader.get_variable_to_dtype_map()
    krnl_mdl_vals = {}

    for k in bst_model.keys():
        if re.match('krnl_mdl*', k):
            # print(k)
            key_str = k.split('/')
            krnl_var_name = str(key_str[1]) #str(key_str[0] + '.' + key_str[1])
            krnl_mdl_vals[krnl_var_name] = tf.convert_to_tensor(ckpt_reader.get_tensor(k))

    try:
        # enum_traj = enumerate(test_data_loader)
        # for batch_idx, batch in enum_traj:
        for batch_idx in range(test_loader.num_batches):
            with file_writer.as_default():
                s_time = time.time()
                gt_traj, future_traj, _ = test_loader.read_batch()
                batch["coords"] = list(gt_traj.values())
                batch["future_coords"] = list(future_traj.values())
                cust_dataloader = {'train_loaderobj': test_loader, 'batch': batch}
                # data_loader = {'train_loaderobj': cust_dataloader, 'batch': batch}
                # train_loader.rl_num_nodes = len(batch['coords'])
                # model = strggrnn(inputs=[args, out_graph, graph, out_sess, data_loader, train_loader.frame_pointer])
                model = strggrnn(inputs=[args, out_graph, graph, cust_dataloader, (batch_idx * args.pred_len), infer])
                model.bst_krnl_mdl_vals = krnl_mdl_vals
                model.fit(infer)
                eager_assess = tf.function(assess_rcmndr)

                tf.config.run_functions_eagerly(True)
                ade_bst, fde_bst, min_idx = eager_assess(args.pred_len,
                                                         model.rl_num_nodes, model.all_pred_path,
                                                         model.nri_obj.target_traj0_ten,
                                                         model=model, n_proposals=model.nri_obj.n_proposals)
                np_ade_bst = ade_bst.numpy()
                np_fde_bst = fde_bst.numpy()
                print('Test ADE = ', np_ade_bst)
                print('Test FDE = ', np_fde_bst)

                # sb.heatmap()
                # make it stateful when batch_size is small enough to find trajectories
                # related to each other between 2 consecutive batches .
                bst_adj_prop = model.nri_obj.adj_mat_vec[min_idx]
                # model.hidden_state = model.krnl_mdl.hidden_states
                model.krnl_mdl.hidden_states = tf.matmul(model.krnl_mdl.attn, model.hidden_state)
                model.krnl_mdl.hidden_states = tf.nn.softmax(model.krnl_mdl.hidden_states)
                num_of_agents = model.rl_num_nodes
                total_ade[0] += np_ade_bst
                total_fde[0] += np_fde_bst
                # preds = torch.tensor(model.ade_op.numpy()[min_idx])
                # # preds = tf.stack(model.all_pred_path.numpy())
                # preds = preds.permute(2, 1, 0)
                # # batch['coords'][:] = preds[:, NUM_OF_PREDS - 20:]
                # t_gt_future = torch.tensor(model.nri_obj.target_traj0_ten.numpy())
                # preds = torch.add(preds[:len(t_gt_future)], t_gt_future)

                # displ = torch.subtract(t_gt_future, preds)
                # total_ade[0] += num_of_agents * tf.reduce_mean(F.pairwise_distance(preds, t_gt_future, p=2)).numpy()

                # total_fde[0] += num_of_agents * tf.reduce_mean(F.pairwise_distance(preds[-1], t_gt_future[-1])).numpy()
                # print('Test ADE = ', total_ade[0] / num_nodes)
                # print('Test FDE = ', total_fde[0] / num_nodes)
                num_nodes += num_of_agents

                tf.summary.scalar(name='ADE', data=np_ade_bst, step=batch_idx)
                print('Running time = ', time.time() - s_time)
                tf.summary.scalar(name='Running Time', data=time.time() - s_time, step=batch_idx)
                tf.summary.scalar(name='RAM used', data=pid.memory_info().rss / 1024 / 1024 / 1024, step=batch_idx)

                fig, ax = matplotlib.pyplot.subplots()
                # fig.canvas.draw()
                bst_adj_mat = bst_adj_prop
                # im = ax.imshow(X=bst_adj_mat, vmin=bst_adj_mat.min(), vmax=bst_adj_mat.max(),
                #                               aspect='auto',
                #                               cmap=matplotlib.pyplot.get_cmap('twilight_shifted'))
                # fig.colorbar(im, cax=ax, orientation='vertical')
                axes = sb.heatmap(data=bst_adj_mat, vmin=bst_adj_mat.min(), vmax=bst_adj_mat.max(),
                                  cmap=matplotlib.pyplot.get_cmap('twilight_shifted'), center=0.5, cbar=True,
                                  square=True)

                # matplotlib.pyplot.plot(np.random.random((10, 10)))
                matplotlib.pyplot.title('Best Adjacency Proposal')
                matplotlib.pyplot.show()

                # img_buf = io.BytesIO()
                # img_buf.seek(0)
                # tf.convert_to_tensor(np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)) #
                fname = summary_logdir + 'bst_adj_mat{}.png'.format(batch_idx)
                fig.savefig(fname)

                # img_buf.seek(0)
                # img_pb = tf.image.decode_png(contents=img_buf.getvalue(), channels=3)
                # img_pb = np.expand_dims(img_pb, 0)
                # tf.summary.image(name='bst_adj_mat', data=img_pb, step=batch_idx)
                # preds_short = preds[:, :args.pred_len]
                # preds_medium = preds[:, :args.pred_len*2:args.pred_frame_hop*2]
                # preds_long = preds[:, :args.pred_len*4:args.pred_frame_hop*4]
        # print('Mean test ADE = ', np.mean(total_ade[0]))
        # print('Mean test FDE = ', np.mean(total_fde[0]))

        print('Mean Test ADE = ', total_ade[0] / test_loader.num_batches)
        print('Mean Test FDE = ', total_fde[0] / test_loader.num_batches)

        file_writer.flush()
        # for i, (scene, agent) in enumerate(zip(batch['scene_id'][:, 0], batch['track_id'][:, 0])):
        #     scene = int(scene.item())
        #     agent = int(agent.item())
        #     if agent in test[str(scene)]:
        #         results[str(scene)][agent] = {}
        #         results[str(scene)][agent]['short'] = preds_short[i].tolist()
        #         results[str(scene)][agent]['medium'] = preds_medium[i].tolist()
        #         results[str(scene)][agent]['long'] = preds_long[i].tolist()

    except TypeError:
        pass

    # with torch.no_grad():
    #     for batch in tqdm(data_loader):
    #         batch = to_gpu(batch)
    #         preds = CNN_Model(batch)
    #         batch['coords'][:] = preds[:, NUM_OF_PREDS - 20:]
    #         preds2 = CNN_Model(batch)
    #         preds = torch.cat((preds, preds2), 1)
    # print('Dump predictions to CNN_Submit.json')
    # with open(parent_dir + 'STR_GGRNN_Submit.json', 'w') as f:
    #     json.dump(results, f)


if __name__ == '__main__':
    main()