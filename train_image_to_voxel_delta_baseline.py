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
import tensorflow as tf
import sys
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('--num-cats-excluded', type=int, default=0)
parser.add_argument('--excluded-cats', type=str, default='')
parser.add_argument('--save-path', type=str)
parser.add_argument('--load-from', type=str, default=None)
parser.add_argument('--concat-features', action='store_true')
parser.add_argument('--use-prior', action='store_true')
# This is a bit awkward boolean below but works
parser.add_argument('--no-delta', action='store_false')
parser.add_argument('--image-code-dim', type=int, default=128)
parser.add_argument('--vox-code-dim', type=int, default=128)
parser.add_argument('--tanh-and-add', action='store_true')
parser.add_argument('--final-sigmoid', action='store_true')
parser.add_argument('--verbosity', type=int, default=0)
args = parser.parse_args()

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



def build_and_train_refiner(load_encoder=True, image_code_dim=16, vox_code_dim=32,
                            num_epochs=1, batches_per_epoch=1, epochs_per_chunk=1,
                            reload_every_chunk=True,
                            regularization=0.0, batch_size=32, vox_encoder_dropout=0.0,
                            image_encoder_dropout=0.0, generator_dropout=0.0, use_batchnorm=False,
                            refiner_path=None, num_test_points=1, empty_prior=False,
                            concat_features=False):
    print("Initializing Refiner")
    print("Intiializing New Encoders")
    im_ae = build_image_ae(image_code_dim, regularization=regularization,
                           dropout_rate=image_encoder_dropout, use_batchnorm=use_batchnorm)
    vox_ae = build_voxel_ae(code_dim=vox_code_dim, regularization=regularization,
                            dropout_rate=vox_encoder_dropout, use_batchnorm=use_batchnorm)
    #im_ae.compile(optimizer=ref_optimizer, loss="binary_crossentropy")
    #vox_ae.compile(optimizer=ref_optimizer, loss="binary_crossentropy")
    vox_ref = delta_refiner.VoxelRefiner(im_ae, vox_ae, regularization=regularization, dropout_rate=generator_dropout,
                                            filter_base_count=32, use_batchnorm=use_batchnorm,
                                          concat_features=concat_features, delta=args.no_delta,
                                        final_sigmoid=args.final_sigmoid,
                                        tanh_and_add = args.tanh_and_add)
    if refiner_path:
        vox_ref.refiner = load_model(refiner_path)
    print("Compiling Refiner")
    vox_ref.refiner.compile(optimizer='Adadelta', loss="binary_crossentropy")
    
    print("Training Refiner")
    train_generator = sampler.triple_generator(category_list=all_categories, n_per_yield=batch_size, mode="train")
    
    category_to_testing_tuple = {}
    category_to_top_score = {}
    for category in all_categories:
        print("Getting Category %s Testing Data" %category)
        cat_generator = sampler.triple_generator(category_list=[category], n_per_yield=num_test_points, mode="val")
        cat_inputs, cat_targets = next(cat_generator)
        category_to_testing_tuple[category] = (cat_inputs, cat_targets)
        category_to_top_score[category] = 0

    top_scores = 0
    top_category_accuracies = 0
    num_epochs_performed = 0
    while num_epochs_performed<num_epochs:
        print("-"*128)
        print("Epoch %d/%.0f" %(num_epochs_performed+1, num_epochs))
        # Get the original data
        vox_ref.refiner.fit_generator(train_generator, steps_per_epoch=batches_per_epoch, epochs=1, verbose=args.verbosity)
        # Now  do testing
        category_current_results = np.zeros(len(all_categories))
        for cat_i, category in enumerate(all_categories):
            [cat_input_images, cat_input_voxels], cat_target_voxels = category_to_testing_tuple[category]
            score = test_iou4(vox_ref.refiner, cat_input_images, cat_input_voxels,cat_target_voxels)
            category_current_results[cat_i] = score
            category_to_top_score[category] = max(score, category_to_top_score[category])
        current_category_accuracies = np.average(category_current_results, axis=0)
        top_category_accuracies = max(top_category_accuracies, current_category_accuracies)
        print("Category-wise Average Performance:", current_category_accuracies, "//Best:", top_category_accuracies)
        num_epochs_performed += 1
        if top_category_accuracies == current_category_accuracies:
            vox_ref.refiner.save(args.save_path)
        sys.stdout.flush()

def test_iou4(refiner_model, train_images, train_voxels, target_voxels):
    num_test_points = len(train_images)
    preds = refiner_model.predict([train_images, train_voxels])
    threshold = 0.4
    pred_ious = round(iou(target_voxels, preds, threshold=threshold),4)
    #top_score = max(top_score, pred_ious[0]) # was .max but I think there's just 1 element
    score = pred_ious
    # print("Threshold", round(threshold,2), ":", pred_ious)
    return score


def main():
    image_code_dim = args.image_code_dim
    vox_code_dim = args.vox_code_dim
    batch_size = 32
    use_batchnorm=False
    num_epochs = np.inf
    batches_per_epoch = 2000
    epochs_per_chunk = 1
    refiner_path = args.load_from
    num_test_points = 1000
    empty_prior = not args.use_prior
    print(empty_prior)
    refiner = build_and_train_refiner(load_encoder=False,
                                      image_code_dim=image_code_dim,
                                      vox_code_dim=vox_code_dim,
                                      batches_per_epoch=batches_per_epoch,
                                      num_epochs=num_epochs,
                                      epochs_per_chunk=epochs_per_chunk,
                                      use_batchnorm=use_batchnorm,
                                      batch_size=batch_size,
                                      refiner_path=refiner_path,
                                      num_test_points=num_test_points,
                                      empty_prior=empty_prior,
                                      concat_features=args.concat_features)
    # refiner_model = load_model('vox_ref_%d_%d.h5' %(image_code_dim, vox_code_dim))
    # load_and_test_refiner(image_code_dim, vox_code_dim)


if __name__ == '__main__':
    main()
