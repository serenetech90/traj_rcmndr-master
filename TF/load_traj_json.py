import numpy as np
import pickle
import csv
import os
import pandas
import glob
import json
import ast
import math
import tensorflow as tf

class DataLoader():
    def __init__(self, args, path, sel=None ,start=0, processFrame=False, infer=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        datasets : The indices of the datasets to use
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        self.parent_dir = '/home/sisi/Documents/Sirius_challenge/Sirius_json/'
        # '/home/serene/PycharmProjects/multimodaltraj_2/data'
        # '/fakepath/Documents/self-growing-spatial-graph/self-growing-gru-offline_avgPool/data'
        # '/Data/stanford_campus_dataset/annotations/'
        # List of data directories where world-coordinates data is stored
        self.used_data_dir = self.parent_dir + 'train/'
        # self.data_dirs = [
        #     parent_dir + '/eth/hotel/',
        #     parent_dir + '/eth/univ/',
        #     parent_dir + '/ucy/zara/zara01/',
        #     parent_dir + '/ucy/zara/zara02/',
        #     parent_dir + '/ucy/zara/zara03/',
        #     parent_dir + '/ucy/univ/',
        #     parent_dir + '/town_center/',
        #     parent_dir + '/annotation_tc.txt'
        #     # parent_dir + '/stanford/bookstore/',
        #     # parent_dir + '/stanford/hyang/',
        #     # parent_dir + '/stanford/coupa/',
        #     # parent_dir + '/stanford/deathCircle/',
        #     # parent_dir + '/stanford/gates/',
        #     # parent_dir + '/stanford/nexus/'
        #     ]
        #self.parent_dir + '/crowds/'
        #self.parent_dir + '/sdd/pedestrians/quad/',
        #self.parent_dir + '/sdd/pedestrians/hyang/',
        #self.parent_dir + '/sdd/pedestrians/coupa/',
        #self.parent_dir + '/sdd/gates/',
        #self.parent_dir + '/sdd/little/',
        #self.parent_dir + '/sdd/deathCircle/'
        #self.parent_dir + '/eth/',
        #self.parent_dir + '/hotel/',
        #self.parent_dir + '/zara/',
        #self.parent_dir + '/crowds/',
        # self.used_data_dirs = [self.data_dirs[x] for x in datasets]
        self.infer = infer

        # Number of datasets
        # self.numDatasets = len(self.data_dirs)

        # Data directory where the pre-processed pickle file resides
        self.data_dir =self.parent_dir
        
        self.traj_batch = {}
        self.targets = {}
        
        # Store the arguments
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.pred_len = args.pred_len
        self.obs_len = args.obs_len
        self.gt_diff = args.gt_frame_hop
        self.pred_diff = args.pred_frame_hop
        self.min_max_coords = [0,0,0,0]
        # Validation arguments
        # self.val_fraction = 0.2
        # Define the path in which the process data would be stored
        # self.current_dir = self.used_data_dirs[start]
        # self.frame_pointer = self.seed
        name = '/trajectories_{0}.cpkl'
        if infer:
            name = '/val_trajectories_{0}.cpkl'

        if os.path.isdir(self.used_data_dir):
            files = self.used_data_dir + "*.json"  # '.' csv
            data_files = sorted(glob.glob(files))
            # if sel is None:
            #     if len(data_files) > 1:
            #         print([x for x in range(len(data_files))])
            #         print(data_files)
            #         self.sel = input('select which file you want for loading:')
            # else:
            #     self.sel = sel
            self.dataset_pointer = sel #str(data_files[int(sel)])[-5]

            self.load_dataset(data_files, val=infer)
            # self.sel_file = self.used_data_dir + name.format(int(self.dataset_pointer))
        else:
            self.dataset_pointer = start  # str(data_files[int(sel)])[-5]
            self.load_dataset(self.used_data_dir, val=infer)
            # self.sel_file = self.current_dir

        # self.sel_file = self.sel_file.split('.')[0] + name.format(int(self.dataset_pointer)) this was added for custom dataset names
        # If the file doesn't exist or forcePreProcess is true
        # processFrame = os.path.exists(self.sel_file)

        # if not processFrame:
        #     print("Creating pre-processed data from raw data")
        #     self.frame_preprocess(self.sel_file, seed=self.seed)

        # Load the processed data from the pickle file
        # self.sel_file =  + name
        # self.load_trajectories(self.sel_file)
        self.num_batches = int((len(self.raw_data)/self.seq_length)/self.batch_size)

    def load_trajectories(self, data_file):
        ''' Load set of pre-processed trajectories from pickled file '''

        f = open(data_file, 'rb')
        self.trajectories = pickle.load(file=f)

        return self.trajectories

    def load_dataset(self, data_files, val=False):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file
        self.raw_data = {}
        self.max_frame = len(data_files[0:100])

        # self.raw_data = np.genfromtxt(fname=data_file, delimiter=',')# remove .transpose()
        csv_json_f = os.path.join(self.parent_dir,'json_to_csv.csv')

        if not os.path.exists(csv_json_f):
            for jfile in data_files[0:100]:
                json_df = open(jfile, 'rb')
                d = json.load(json_df)
                df = pandas.DataFrame(d["AGENTS"])
                for k in df.keys():
                    if k in self.raw_data.keys():
                        self.raw_data[k].extend(df[k]["coords"])
                    else:
                        self.raw_data[k] = df[k]["coords"]
                # print("Completed reading JSON file {}".format(json_df.name))
                json_df.close()

            self.tr_data = self.raw_data
            jsonf = open(csv_json_f, 'w')
            json_wr = csv.writer(jsonf, delimiter=',')
            json_wr.writerows(list(self.tr_data.items()))
            jsonf.close()
        else:
            self.raw_data = dict({str(k):ast.literal_eval(v) for (k,v) in pandas.read_csv(csv_json_f, delimiter=',').values})

        self.tr_data = self.raw_data
        self.pedIDs = self.raw_data.keys()
        self.itr_pedIDS = next(iter(self.pedIDs))


    def read_batch(self):
        '''
        Function to get the next batch of trajectories
        '''
        # Source data
        
        b = -1

        while b < self.batch_size:
            b += 1
            for id in self.pedIDs:
                try:
                    gt_obs_traj = self.tr_data[id][(b*self.frame_pointer):(b*self.frame_pointer)+self.obs_len]
                    if len(gt_obs_traj) > 0:
                        if int(id) not in self.traj_batch.keys():
                            self.traj_batch.update({int(id): np.array(gt_obs_traj, dtype=object, ndmin=2)})
                        else:
                            # print(self.traj_batch[int(id)].shape, gt_obs_traj.shape, id)
                            self.traj_batch[int(id)] = np.concatenate((self.traj_batch[int(id)], gt_obs_traj), axis=0)

                except KeyError:
                    break
            self.tick_frame_pointer(hop=self.gt_diff)
            # self.reset_data_pointer(frame_pointer=self.gt_diff)
            for idx_c in self.traj_batch.keys():
                gt_future_traj = self.tr_data[str(idx_c)][self.frame_pointer:self.frame_pointer + self.pred_len]
                if len(gt_future_traj) > 0:
                    if len(self.targets) == 0:
                        self.targets = {idx_c: np.array(gt_future_traj, dtype=object, ndmin=2)}
                    elif idx_c not in self.targets:
                        self.targets.update({idx_c: np.array(gt_future_traj, dtype=object, ndmin=2)})
                    else:
                        self.targets[idx_c] = np.concatenate([self.targets[idx_c], gt_future_traj], axis=0)
            self.tick_frame_pointer(hop=self.pred_diff)

        self.rl_num_nodes = len(self.traj_batch.keys())

        return self.traj_batch, self.targets, self.frame_pointer

    def tick_frame_pointer(self, hop, valid=False):
        '''
        Advance the frame pointer
        '''
        if not valid:
            self.frame_pointer += hop

    def reset_data_pointer(self, valid=False, dataset_pointer=0, frame_pointer=0):
        '''
        Reset all pointers
        '''
        if not valid:
            self.frame_pointer = frame_pointer
        else:
            self.dataset_pointer = dataset_pointer
            self.frame_pointer = frame_pointer
