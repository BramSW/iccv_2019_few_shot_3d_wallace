import numpy as np
from keras import models
import highres_sampler
from iou import iou
import matplotlib.pyplot as plt
from pprint import pprint
import pickle
import argparse
import tensorflow as tf
import random
from copy import copy


all_categories = ["airplanes",
	      "benches",
	      "cabinets",
	      "cars",
	      "chairs",
	      "displays",
	      "lamps",
	      "phones",
	      "rifles",
	      "sofas",
	      "speakers",
	      "tables",
	      "vessels"]

category_to_input_images = {}
category_to_input_voxels = {}
category_to_target_voxels = {}

num_test_points = 1000
for mode in ['train', 'val', 'test']:
    for category in all_categories:
        print("Getting Category %s Testing Data" %category)
        cat_generator = highres_sampler.triple_generator(category_list=[category], n_per_yield=num_test_points, mode=mode)
        cat_inputs, cat_targets = next(cat_generator)
        category_to_input_images[category] = cat_inputs[0]
        category_to_input_voxels[category] = cat_inputs[1]
        category_to_target_voxels[category] = cat_targets

    pickle.dump(category_to_input_images, open('RGB_' + mode + '_input_images.pickle', 'wb'))
    pickle.dump(category_to_input_voxels, open('RGB_' + mode + '_input_voxels.pickle', 'wb'))
    pickle.dump(category_to_target_voxels, open('RGB_' + mode + '_target_voxels.pickle', 'wb'))

