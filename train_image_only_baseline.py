from keras.models import load_model
#import refiner
import delta_refiner
from binvox_rw import read_as_3d_array
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
import os
import pickle
from iou import iou
import sampler
from image_autoencoder import build_image_ae
from voxel_autoencoder import build_voxel_ae
import random 
import argparse
from copy import copy 
from keras import layers
import tensorflow as tf
from keras.models import Sequential
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


def test_iou4(refiner_model, train_images, target_voxels):
    num_test_points = len(train_images)
    preds = refiner_model.predict(train_images)
    threshold = 0.4
    pred_ious = round(iou(target_voxels, preds, threshold=threshold),4)
    #top_score = max(top_score, pred_ious[0]) # was .max but I think there's just 1 element
    score = pred_ious
    # print("Threshold", round(threshold,2), ":", pred_ious)
    return score

parser = argparse.ArgumentParser()
parser.add_argument('--num-cats-excluded', type=int, default=0)
parser.add_argument('--excluded-cats', type=str, default='')
parser.add_argument('--save-path', type=str)
parser.add_argument('--load-from', type=str, default=None)
parser.add_argument('--concat-features', action='store_true')
parser.add_argument('--use-prior', action='store_true')
parser.add_argument('--no-delta', action='store_false')
parser.add_argument('--num-refine-iters', type=int)
parser.add_argument('--image-code-dim', type=int, default=128)
parser.add_argument('--vox-code-dim', type=int, default=128)
parser.add_argument('--final-sigmoid', action='store_true')
parser.add_argument('--verbosity', type=int, default=0)
args = parser.parse_args()
num_refine_iters = args.num_refine_iters

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
                      "vessels"][args.num_cats_excluded:]

if args.excluded_cats:
    excluded_cats_list = args.excluded_cats.split(',')
    for cat in excluded_cats_list:
        all_categories.remove(cat)


image_code_dim = args.image_code_dim
batch_size = 32
num_epochs = np.inf
batches_per_epoch = 2000
load_from = args.load_from
num_test_points = 1000

kernel_size=(5,5,5)
filter_base_count=32
strides=(2,2,2)
  
if load_from:
    model = load_model(load_from)
else:
    model = Sequential()
    model.add(layers.Conv2D(kernel_size=(5,5), strides=(2,2), filters=128, activation='relu',
                         input_shape=(32,32,1)))
    model.add(layers.Conv2D(kernel_size=(5,5), strides=(2,2), filters=256, activation='relu'))
    model.add(layers.Conv2D(kernel_size=(5,5), strides=(2,2), filters=512, activation='relu'))
    model.add(layers.Reshape((512,)))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(image_code_dim, activation='relu', name='im_code'))



    model.add(layers.Dense(128*4*4*4, activation='relu'))
    model.add(layers.Reshape((4, 4, 4, 128)))
    model.add(layers.Conv3DTranspose(4*filter_base_count,
                                        kernel_size,
                                        strides=strides,
                                        padding='same',  
                                        activation='relu'))
    model.add(layers.Conv3DTranspose(2*filter_base_count,
                                        kernel_size,
                                        strides=strides,
                                        padding='same',  
                                        activation='relu'))
    model.add(layers.Conv3DTranspose(1,
                                    kernel_size,
                                    strides=strides,
                                    padding='same',  
                                    activation='sigmoid'))

    model.compile(optimizer='Adadelta', loss="binary_crossentropy")


def single_input_generator(category_list=all_categories, n_per_yield=batch_size, mode="train", yield_categories=[]):
    internal_generator = sampler.triple_category_generator(category_list=category_list,
                                                           n_per_yield=n_per_yield,
                                                           mode=mode,
                                                           yield_categories=yield_categories)
    while True:
        input_tuple, target = next(internal_generator)
        yield (input_tuple[0], target)


print("Training Refiner")

train_generator = single_input_generator(category_list=all_categories, n_per_yield=batch_size, mode="train")

category_to_testing_tuple = {}
category_to_top_score = {}
for category in all_categories:
    print("Getting Category %s Testing Data" %category)
    cat_generator = single_input_generator(category_list=all_categories, n_per_yield=num_test_points, mode="val",
                                                     yield_categories=[category])
    cat_inputs, cat_targets = next(cat_generator)
    category_to_testing_tuple[category] = (cat_inputs, cat_targets)
    category_to_top_score[category] = 0

top_score = 0
top_category_accuracies = 0
num_epochs_performed = 0

while num_epochs_performed<num_epochs:
    print("-"*128)
    print("Epoch %d/%.0f" %(num_epochs_performed+1, num_epochs))
    # Get the original data
    # Do the training for each iteration
    model.fit_generator(train_generator, steps_per_epoch=batches_per_epoch, epochs=1, verbose=args.verbosity)
    # Now  do testing
    category_current_results = [0] * len(all_categories)
    for cat_i, category in enumerate(all_categories):
        cat_input_images, cat_target_voxels = category_to_testing_tuple[category]
        print("Test Performance %s:" %category.upper())
        score = test_iou4(model, cat_input_images ,cat_target_voxels)
        category_current_results[cat_i] = np.append(category_current_results[cat_i], score)
        category_to_top_score[category] = max(score, category_to_top_score[category])
    current_category_accuracies = np.average(category_current_results)
    top_category_accuracies = max(top_category_accuracies, current_category_accuracies)
    print("Category-wise Average Performance:", current_category_accuracies, "//Best:", top_category_accuracies)
    num_epochs_performed += 1
    if top_category_accuracies == current_category_accuracies:
        model.save(args.save_path)
