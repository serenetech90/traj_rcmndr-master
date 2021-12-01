import glob
import io
import numpy as np
import tensorboardX as tbx
import seaborn as sb
from tensorflow import compat as tf1

parent_dir = '/home/sisi/Documents/Sirius_challenge/'
logdir = parent_dir + 'traj_rcmndr-master/TF/data/train/TBX/zara1_test/'
logs = glob.glob(logdir+'/events*')

for evefile in logs:
    event = tf1.v1.train.summary_iterator(evefile)
    for field in event:
        field_prop = field.summary.Value
        if field_prop.tag in ['RAM used', 'Running Time', 'ADE']:
            img = tf1.v1.io.decode_image(contents=field_prop.image.encoded_image_string, name='zara1_frame')