import multiprocessing
import random
import numpy as np
import pickle
import csv
import os
import json
import threading
from multiprocessing import Process
from multiprocessing import Pool
from matplotlib.pyplot import imread
from glob import glob

# import torch
import tensorflow as tf

def rang(x):
    return x-1

def str2int(x):
    return int(float(x))

# def _read(flstruct):
#     (frameList, pedsPerFrameList, x) = flstruct
#     for frame_pointer in frameList:
#         for (ind, ped, pos_x, pos_y) in pedsPerFrameList:
#             if ind == frame_pointer:
#                 try:
#                     x[ped].append([pos_x, pos_y])
#                 except KeyError:
#                     x[ped] = [pos_x, pos_y]

class DataLoader():
    def __init__(self, args, path, leave=0 ,start=0, processFrame=False, infer=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        datasets : The indices of the datasets to use
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        self.parent_dir = '/home/sisi/Documents/Sirius_challenge/traj_rcmndr-master/TF/data/'
        # '/home/serene/PycharmProjects/multimodaltraj_2/data'
        # '/fakepath/Documents/self-growing-spatial-graph/self-growing-gru-offline_avgPool/data'
        # '/Data/stanford_campus_dataset/annotations/'
        # List of data directories where world-coordinates data is stored
        if infer:
            self.used_data_dir = self.parent_dir + 'train/' + "{0}.csv".format(leave)
            self.trajfilename = self.parent_dir + 'train/' + '/val_ggrnnv_trajectories_{0}_0.cpkl'.format(leave)
        else:
            self.used_data_dir = self.parent_dir + 'train/'
            self.trajfilename = self.used_data_dir + '/ggrnnv_trajectories_{0}_1.cpkl'.format(leave)
            files = self.used_data_dir + "*.csv"  # '*.json'
            data_files = sorted(glob(files))
            data_files.remove(self.used_data_dir + str(leave) + '.csv')
            # by default remove non-vislet datasets
            data_files.remove(self.used_data_dir + '0.csv')
            data_files.remove(self.used_data_dir + '1.csv')

        self.infer = infer
        self.num_workers = 6
        # Number of datasets
        # self.numDatasets = len(self.data_dirs)

        # Data directory where the pre-processed pickle file resides
        self.data_dir =self.parent_dir
        
        self.traj_batch = {}
        self.traj_vislet = {}
        self.targets = {}
        self.img_key = []
        self.snapshot = {}


        # Store the arguments
        self.args = args
        self.lambda_param = args.lambda_param
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.pred_len = args.pred_len
        self.obs_len = args.obs_len
        self.gt_diff = args.gt_frame_hop
        self.pred_diff = args.pred_frame_hop
        self.min_max_coords = [0,0,0,0]
        self.dim = int(args.neighborhood_size / args.grid_size)
        self._2dconv_in = tf.zeros(shape=(self.dim, self.dim))
        # Validation arguments
        # self.val_fraction = 0.2
        # Define the path in which the process data would be stored
        # self.current_dir = self.used_data_dirs[start]
        # self.frame_pointer = self.seed

        self.dataset_pointer = start
        # data_files = self.used_data_dir + "{0}.csv".format(leave)  # '*.json'
        # data_files = sorted(glob.glob(files))
        # data_files.remove(self.used_data_dir + str(leave) + '.csv')

        if os.path.isdir(self.used_data_dir):
            # if sel is None:
            #     if len(data_files) > 1:
            #         print([x for x in range(len(data_files))])
            #         print(data_files)
            #         self.sel = input('select which file you want for loading:')
            # else:
            #     self.sel = sel
             #str(data_files[int(sel)])[-5]
            self.load_dataset(data_files, val=infer)
        else:
            self.load_dataset(self.used_data_dir, val=infer)

            # self.sel_file = self.current_dir
        self.sel_file = self.trajfilename
        # self.sel_file = self.sel_file.split('.')[0] + name.format(int(self.dataset_pointer)) this was added for custom dataset names
        # If the file doesn't exist or forcePreProcess is true
        processFrame = os.path.exists(self.sel_file)

        if not processFrame:
            print("Creating pre-processed data from raw data")
            self.frame_preprocess()
        self.load_trajectories(self.sel_file)
        # Load the processed data from the pickle file
        # self.sel_file =  + name

    def frame_preprocess(self, data_files=None, seed=0, infer=False):

        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''

        frame_data = {i: {} for i in self.frameList}
        self.pedsPerFrameList = np.transpose(self.pedsPerFrameList)
        x = {}

        def _read(p, s, e, return_dict):
            for frame_pointer, ele in enumerate(self.frameList[s:e]):
                # print('Dataset frames snippet read between frames {} and {}'.format(minf, maxf))
                for (ind, ped, pos_x, pos_y, v_x, v_y) in self.pedsPerFrameList[s:e]:
                    if ind == ele:
                        if ped not in x.keys():
                            x[ped] = {}
                        try:
                            x[ped]['gt_traj'].append([pos_x, pos_y])
                            x[ped]['gt_vis'].append([v_x, v_y])
                        except KeyError:
                            x[ped]['gt_traj'] = [[pos_x, pos_y]]
                            x[ped]['gt_vis'] = [[v_x, v_y]]

                    # return_dict['x_' + str(int(ele))] = x
                    # if not 'x_'+str(int(ele)) in return_dict.keys():
                    #     return_dict['x_'+str(int(ele))] = x
                    # # else:
                    # #     return_dict['x_' + str(int(ele))] = {'x': x}
                    # elif not ped in return_dict['x_' + str(int(ele))].keys():
                    #     return_dict['x_' + str(int(ele))].update({ped: x[ped]})
                    # else:
                    #     return_dict['x_' + str(int(ele))][ped].append(x[ped])

                # print('Frame count {}, Read index = {}, and Max frame is {}, at Process {}'.format(frame_pointer, ele, maxf, p))

            return_dict['x'] = x
            print('Process {} finished'.format(p))

        def _write(p, s, e, minf, maxf, return_dict):
            for frame_pointer, ele in enumerate(self.frameList[s:e]):
                print('Dataset frames snippet write from {} to {} at Process {}'.format(minf, maxf, p))
                if len(frame_data[ele]) > 0:
                    # if len(frame_data[self.frame_pointer][ind]) > 0:
                    for idx in return_dict['x']:
                        try:
                            frame_data[ele][idx].append(return_dict['x'][idx])
                        except KeyError:
                            frame_data[ele][idx] = return_dict['x'][idx]
                else:
                    try:
                        frame_data[ele] = return_dict['x']
                    except KeyError:
                        pass
                print('Frame count {}, Write index = {}, and Max frame is {}, at Process {}'.format(frame_pointer, ele, maxf, p))

            print('Process {} finished'.format(p))
            return_dict['frame_data_'+str(int(s))+'_'+str(int(e))] = frame_data

        e = int(self.max / self.num_workers)
        # processes = self.pool.map(func=rang, iterable=range(12))

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        pthreads = [] * self.num_workers  # len(processes)
        for k in range(self.num_workers):
            # tf.pad(nri_obj.adj_mat_vec[k], paddings=[[1,1], [1,1]])
            # print('k = ', k)
            print('Started Reading Thread = ', k)
            # minf = self.frameList[k*e:(k*e)+e].min()
            # maxf = self.frameList[k*e:(k*e)+e].max()
            pthreads.append(Process(target=_read, args=(k, k*e, (k*e)+e, return_dict)))
            pthreads[k].start()
            # pthreads[k].join()

        for p in pthreads:
            p.join()

        # f = open(self.used_data_dir + 'peds_dict.txt', 'w')
        # f.write(str(return_dict))
        # f.close()
        #
        # pthreads = [] * self.num_workers  # len(processes)
        # # for k, _ in enumerate(processes):
        # for k in range(self.num_workers):
        #     # tf.pad(nri_obj.adj_mat_vec[k], paddings=[[1,1], [1,1]])
        #     # print('k = ', k)
        #     minf = self.frameList[k*e:(k*e)+e].min()
        #     maxf = self.frameList[k*e:(k*e)+e].max()
        #     print('Started Writing Thread = ', k)
        #     pthreads.append(Process(target=_write, args=(k, k*e, (k*e)+e, minf, maxf, return_dict)))
        #     pthreads[k].start()
        #     # self.tick_frame_pointer(hop=self.gt_diff)
        #
        # for p in pthreads:
        #     p.join()

        f = open(self.trajfilename, "w")
        # pickle.dump(return_dict, f, protocol=2)
        json.dump(dict(return_dict), f)
        f.close()

    def load_trajectories(self, data_file):
        ''' Load set of pre-processed trajectories from pickled file '''

        f = open(data_file, 'r')
        self.trajectories = json.load(f)

        return self.trajectories

    def load_dataset(self, data_files, val=False):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file
        self.raw_data = {}
        self.tr_data = []

        if not val:
            for idx_f, f in enumerate(data_files):
                self.img_key.append(f.split('.')[0][-1])
                ctxt_img = glob(self.parent_dir + '/train/' + 'ctxt_{}.png'.format(self.img_key[-1]))[0]
                ctxt_img = tf.convert_to_tensor(imread(ctxt_img), dtype=tf.float32)

                ctxt_img_pd = tf.convert_to_tensor(tf.pad(ctxt_img, paddings=tf.constant([[1, 1, ], [0, 1], [0, 0]])),
                                                   dtype=tf.float32)

                ctxt_img_pd = tf.expand_dims(ctxt_img_pd, axis=0)
                width = ctxt_img_pd.shape[1]
                height = ctxt_img_pd.shape[2]
                if ctxt_img_pd.shape[3] == 3:
                    _2dconv = tf.nn.conv2d(input=ctxt_img_pd,
                                           filters=tf.keras.initializers.truncated_normal()(
                                               shape=[width - self.dim + 1,
                                                      height - self.dim + 1, 3, 1],
                                               dtype=tf.float32),
                                       padding='VALID', strides=[1, 1, 1, 1])
                elif ctxt_img_pd.shape[3] == 4:
                    _2dconv = tf.nn.conv2d(input=ctxt_img_pd,
                                           filters=tf.keras.initializers.truncated_normal()(
                                               shape=[width - self.dim + 1,
                                                      height - self.dim + 1, 4, 1],
                                               dtype=tf.float32),
                                       padding='VALID', strides=[1, 1, 1, 1])

                _2dconv = tf.squeeze(_2dconv)
                # _2dconv = self.lambda_param * _2dconv

                self.snapshot[self.img_key[-1]] = {'snapshot': _2dconv, 'snapshot_fname': ctxt_img_pd, 'idx_read': [], 'width': width, 'height': height}
                self._2dconv_in += _2dconv

                f = open(f, 'rb')
                self.raw_data[idx_f] = np.genfromtxt(fname=f, delimiter=',')  # remove
                f.close()

            rawOD = self.raw_data
            keys = list(rawOD.keys())
            keys = random.sample(keys, len(keys))
            for k in keys:
                self.raw_data[k] = rawOD[k][0:6]

            for _, k in enumerate(keys):
                if _ == 0:
                    self.tr_data = self.raw_data[k]
                else:
                    self.tr_data = np.concatenate((self.tr_data, self.raw_data[k]), axis=1)

        else:
            f = open(data_files, 'rb')
            self.raw_data = np.genfromtxt(fname=f, delimiter=',')  # remove
            f.close()
            self.tr_data = self.raw_data[0:6]
            self.img_key.append(str(self.args.leaveDataset))
            ctxt_img = glob(self.parent_dir + '/train/' + 'ctxt_{}.png'.format(self.img_key[-1]))[0]
            ctxt_img = tf.convert_to_tensor(imread(ctxt_img), dtype=tf.float32)

            ctxt_img_pd = tf.convert_to_tensor(tf.pad(ctxt_img, paddings=tf.constant([[1, 1, ], [0, 1], [0, 0]])),
                                               dtype=tf.float32)

            ctxt_img_pd = tf.expand_dims(ctxt_img_pd, axis=0)
            width = ctxt_img_pd.shape[1]
            height = ctxt_img_pd.shape[2]
            if ctxt_img_pd.shape[3] == 3:
                _2dconv = tf.nn.conv2d(input=ctxt_img_pd,
                                       filters=tf.keras.initializers.random_normal()(
                                           shape=[width - self.dim + 1,
                                                  height - self.dim + 1, 3, 1],
                                           dtype=tf.float32),
                                       padding='VALID', strides=[1, 1, 1, 1])
            elif ctxt_img_pd.shape[3] == 4:
                _2dconv = tf.nn.conv2d(input=ctxt_img_pd,
                                       filters=tf.keras.initializers.random_normal()(
                                           shape=[width - self.dim + 1,
                                                  height - self.dim + 1, 4, 1],
                                           dtype=tf.float32),
                                       padding='VALID', strides=[1, 1, 1, 1])

            _2dconv = tf.squeeze(_2dconv)
            _2dconv += self.lambda_param * _2dconv

            self.snapshot[self.img_key[-1]] = {'snapshot': _2dconv, 'snapshot_fname': ctxt_img_pd, 'idx_read': [],
                                               'width': width, 'height': height}
            self._2dconv_in += _2dconv
            self._2dconv_in = self.snapshot[str(self.args.leaveDataset)]['snapshot']

        self._2dconv_in += tf.expand_dims(
            tf.range(start=0, limit=1, delta=(1 / self.args.obs_len), dtype=tf.float32), axis=0)
        # Get all the data from the pickle file
        # self.data = self.raw_data[:,2:4]

        # Randomize order of trajectories across the datasets
        self.len = len(self.tr_data)
        self.max = int(self.tr_data.shape[1] * 0.7)  #
        self.val_max = int(self.tr_data.shape[1] * 0.3)
        self.val_data = self.tr_data[:, self.max:self.max + self.val_max]
        self.tr_data = self.tr_data[:, 0:self.max]

        if not val:
            self.frameList = self.tr_data[0, :]

            self.pedsPerFrameList = self.tr_data[0:6, :]
            self.vislet = self.tr_data[4:6, :]

            self.seed = self.frameList[0]
            self.frame_pointer = int(self.seed)
            self.num_batches = int((self.max / self.seq_length) / self.batch_size)
        else:
            self.frameList = self.val_data[0, :]

            self.pedsPerFrameList = self.val_data[0:6, :]
            self.vislet = self.val_data[4:6, :]

            self.seed = self.frameList[0]
            self.frame_pointer = 0 #int(self.seed)

            self.num_batches = int((self.val_max / self.seq_length) / self.batch_size)

        self.max_frame = max(self.frameList)
        self.pedIDs = self.pedsPerFrameList[1, :]
        self.itr_pedIDS = next(iter(self.pedIDs))

        # csv_json_f = os.path.join(self.parent_dir,'json_to_csv.csv')

        # if not os.path.exists(csv_json_f):
        #     for jfile in data_files[0:100]:
        #         json_df = open(jfile, 'rb')
        #         d = json.load(json_df)
        #         df = pandas.DataFrame(d["AGENTS"])
        #         for k in df.keys():
        #             if k in self.raw_data.keys():
        #                 self.raw_data[k].extend(df[k]["coords"])
        #             else:
        #                 self.raw_data[k] = df[k]["coords"]
        #         # print("Completed reading JSON file {}".format(json_df.name))
        #         json_df.close()
        # jsonf = open(csv_json_f, 'w')
        # json_wr = csv.writer(jsonf, delimiter=',')
        # json_wr.writerows(list(self.tr_data.items()))
        # jsonf.close()
        # else:
        #     self.raw_data = dict({str(k):ast.literal_eval(v) for (k,v) in pandas.read_csv(csv_json_f, delimiter=',').values})

        # self.pedIDs = self.raw_data.keys()

    def read_batch(self, infer=False):

        '''
        Function to get the next batch of trajectories
        '''
        # Source data
        b = -1
        while b < self.batch_size:
            b += 1
            # for pid in self.trajectories:
            for pid, ptraj in self.trajectories['x'].items():
                try:
                    # pid = str2int(pid)
                    gt_obs_traj = tf.convert_to_tensor(ptraj['gt_traj'][(b * self.frame_pointer):(b * self.frame_pointer) + self.obs_len], dtype=tf.float32)
                    gt_vislets = tf.convert_to_tensor(ptraj['gt_vis'][(b * self.frame_pointer):(b * self.frame_pointer) + self.obs_len], dtype=tf.float32)
                    # gt_obs_traj = self.tr_data[pid][(b * self.frame_pointer):(b * self.frame_pointer) + self.obs_len]
                    # if len(gt_obs_traj) > 0:
                    if gt_obs_traj.shape[0] > 0:
                        if pid not in self.traj_batch.keys():
                            self.traj_batch.update({pid: gt_obs_traj})
                            self.traj_vislet.update({pid: gt_vislets})
                        else:
                            # print(self.traj_batch[int(pid)].shape, gt_obs_traj.shape, pid)
                            self.traj_batch[pid] = tf.concat((self.traj_batch[pid], gt_obs_traj), axis=0)
                            self.traj_vislet[pid] = tf.concat((self.traj_vislet[pid], gt_vislets), axis=0)

                except KeyError:
                    pass

            self.tick_frame_pointer(hop=self.gt_diff)
            # self.reset_data_pointer(frame_pointer=self.gt_diff)
            for idx_c in self.traj_batch.keys():
                # gt_future_traj = self.tr_data[idx_c][self.frame_pointer:self.frame_pointer + self.pred_len]
                # try:
                gt_future_traj = tf.convert_to_tensor(self.trajectories['x'][idx_c]['gt_traj'][self.frame_pointer:self.frame_pointer + self.pred_len],
                                                      dtype=tf.float32)
                # except IndexError:
                #     end = len(self.trajectories['x'][idx_c]) - 1
                #     gt_future_traj = np.concatenate(self.trajectories['x'][idx_c][self.frame_pointer:self.frame_pointer + end],
                #                                     np.zeros(abs(end-self.pred_len)), axis=0)
                # gt_future_traj = self.tr_data[str(idx_c)][self.frame_pointer:self.frame_pointer + self.pred_len]
                # if len(gt_future_traj) > 0:
                if gt_future_traj.shape[0] > 0:
                    if len(self.targets) == 0:
                        self.targets = {idx_c: gt_future_traj}
                    elif idx_c not in self.targets:
                        self.targets.update({idx_c: gt_future_traj})
                    elif self.targets[idx_c].shape[0] < self.pred_len:
                        self.targets[idx_c] = tf.concat((self.targets[idx_c], gt_future_traj), axis=0)
            self.tick_frame_pointer(hop=self.pred_diff)

        self.rl_num_nodes = len(self.traj_batch.keys())

        return self.traj_batch, self.targets, self.frame_pointer

    def tick_frame_pointer(self, hop, valid=False):
        '''
        Advance the frame pointer
        '''
        if not valid:
            self.frame_pointer += hop

    def reset_data_pointer(self, valid=False, dataset_pointer=0, frame_pointer=None):
        '''
        Reset all pointers
        '''
        # if not valid:
        #     self.frame_pointer = frame_pointer
        # else:
        self.dataset_pointer = dataset_pointer
        if frame_pointer:
            self.frame_pointer = frame_pointer

