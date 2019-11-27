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
plt.rcParams['font.size'] = 14.0

randomize_input_cats = True

num_test_points = 100
num_iters = 10 

transfer_cats = ["benches",
                 "cabinets",
                  "lamps",
                  "rifles",
                  "sofas",
                   "vessels"]


num_batches = num_test_points // 100

pickle_string = 'iterative_performances_randomized_inputs.pickle'
if os.path.exists(pickle_string):
    label_to_info_hash = pickle.load(open(pickle_string, 'rb'))
    print(label_to_info_hash)
else:
    label_to_info_hash = {}

model_1_iter_1_shot = None #.load_model('models/1_iter_1_shot.h5')
model_2_iter_1_shot = None #.load_model('models/2_iter_1_shot.h5')
model_1_iter = None #.load_model('models/1_iter.h5')
model_2_iter = None #.load_model('models/2_iter.h5')
model_3_iter = None #.load_model('models/3_iter.h5')


model_prior_tuples = [ (model_1_iter, False, 1, "1 Iteration Full Prior Training: 1-Shot", transfer_cats, 'o'),
                           (model_2_iter, False, 1, "2 Iterations Full Prior Training: 1-Shot", transfer_cats, '^'),
                           (model_3_iter, False, 1, "3 Iterations Full Prior Training: 1-Shot", transfer_cats, 's'),
                           (model_1_iter_1_shot, False, 1, "1 Iteration 1-shot Prior Training: 1-Shot", transfer_cats, 'P'),
                           (model_2_iter_1_shot, False, 1, "2 Iteration 1-shot Prior Training: 1-Shot", transfer_cats, 'X')]



plt.figure()

for model, input_average_voxel, k_shot_prior, label, category_list, marker in model_prior_tuples:
    print(label)
    if label not in label_to_info_hash:
        category_to_score_arr = {cat:[] for cat in category_list}
        for category in category_list:
            generator = highres_sampler.full_test_generator(category_list=[category], n_per_yield=100, mode="test", input_average_voxel=input_average_voxel,
                                                     k_shot_prior=k_shot_prior, randomize_categories=randomize_input_cats)
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
        #Could below be doing something wierd?
        print(list(category_to_score_arr.values()))
        averaged_iter_performances = np.average(np.array(list(category_to_score_arr.values())), axis=0)
        label_to_info_hash[label] = averaged_iter_performances
    else:
        averaged_iter_performances = label_to_info_hash[label]
    plt.plot(range(1, num_iters+1), averaged_iter_performances, label=label)
plt.xlabel("# Iterations")
plt.ylabel("IoU")
# plt.axhline(y=0.631, linestyle='dashed', label='Training Baseline', color='r')
plt.ylim(0.175, 0.375)
plt.axhline(y=0.361, linestyle='dashed', label='Transfer Baseline', color='g')
plt.legend(fontsize=12, loc='lower right')
plt.tight_layout()

pickle.dump(label_to_info_hash, open(pickle_string, 'wb')) 

image_save_string = 'randomized_inputs_iterative_peformance.png' if randomize_input_cats else 'iterative_performance.png'
plt.savefig(image_save_string)



