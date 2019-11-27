import numpy as np
from keras import models
from iou import iou
import matplotlib.pyplot as plt
from pprint import pprint
import pickle
import argparse
import tensorflow as tf
import random
from copy import copy
from keras.backend.tensorflow_backend import set_session
import highres_sampler
import os
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12.0


num_test_points = 100
num_iters = 10

gy_cats        = ["airplanes",
                  "cars",
                  "chairs",
                  "displays",
                  "phones",
                  "speakers",
                  "tables"]

transfer_cats = ["benches",
                 "cabinets",
                  "lamps",
                  "rifles",
                  "sofas",
                   "vessels"]


num_batches = num_test_points // 100

pickle_string = "categorical_iterative_performances.pickle"
if os.path.exists(pickle_string):
    label_to_info_hash = pickle.load(open(pickle_string, 'rb'))
    print(label_to_info_hash)
else:
    label_to_info_hash = {}

model_3_iter = None # models.load_model('/data/bw462/3d_recon/models/3_iter.h5')


model_prior_tuples = [(model_3_iter, True, 0, "3 Iterations Full Prior Training: Full Prior", transfer_cats, 's')] #,
#                             (model_3_iter, False, 1, "3 Iterations Full Prior Training: 1-Shot", transfer_cats, 's')]


fig, axes = plt.subplots(1,1,sharey=True, sharex=True) ###


##ax = axes[0]
lines = [[], []]
labels = [[], []]
for model, input_average_voxel, k_shot_prior, label, category_list, marker in model_prior_tuples:
    print(label)
    if label not in label_to_info_hash:
        category_to_score_arr = {cat:[] for cat in category_list}
        for category in category_list:
            generator = highres_sampler.full_test_generator(category_list=[category], n_per_yield=100, mode="test", input_average_voxel=input_average_voxel,
                                                     k_shot_prior=k_shot_prior)
            #generator = highres_sampler.triple_generator(category_list=[category], n_per_yield=10, mode="test", input_average_voxel=input_average_voxel,
            #                                         k_shot_prior=k_shot_prior)
            all_batch_ious = []
            for [input_im, input_vox], target_vox in generator:
                batch_ious = []
                for iter_i in range(num_iters):
                    preds = model.predict([input_im, input_vox])
                    iou_score = iou(preds, target_vox, threshold=0.4)
                    batch_ious.append(round(iou_score,3))
                    input_vox = preds
                all_batch_ious.append(batch_ious)
            iteration_performances = np.average(all_batch_ious, axis=0)
            category_to_score_arr[category] = iteration_performances
            print(category, iteration_performances)
        print(category_to_score_arr)
        label_to_info_hash[label] = category_to_score_arr
    else:
        category_to_score_arr = label_to_info_hash[label]
    ax_index = 0 if label[-6:] == '1-Shot' else 1
    ax = axes###[ax_index]
    for category in category_list:
        #if ax_index ==0:
        #    label = category if category in ['benches', 'cabinets', 'lamps'] else None
        #if ax_index ==1:
        #    label = category if category in ['rifles', 'sofas', 'vessels'] else None
        line = ax.plot(range(1, num_iters+1), category_to_score_arr[category][:num_iters], label=category, linewidth=3)#, marker=marker, linestyle="None")

ax.set_title("Full-Shot Prior Input")
ax.axhline(y=0.361, linestyle='dashed', color='g')
ax.set_xlabel("# Iterations")
ax.set_ylabel("IoU")
ax.legend(loc="upper right")
"""
axes[0].set_title("1-Shot Prior Input")
axes[1].set_title("Full Prior Input")
axes[0].axhline(y=0.361, linestyle='dashed', color='g')
axes[1].axhline(y=0.361, linestyle='dashed', label='Transfer Baseline', color='g')
axes[0].set_xlabel("# Iterations")
axes[1].set_xlabel("# Iterations")
axes[0].set_ylabel("IoU")
axes[1].set_ylabel("IoU")
axes[0].legend(loc="upper right")
axes[1].legend(loc="lower right")
"""
plt.tight_layout()

#plt.show()
image_save_string = 'categorical_iterative_performance.png'
plt.savefig(image_save_string)

pickle.dump(label_to_info_hash, open(pickle_string, 'wb')) 


