from keras.models import load_model, Sequential
#import refiner
import delta_refiner
from binvox_rw import read_as_3d_array
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
import os
import pickle
from iou import iou
import highres_sampler
from image_autoencoder import build_image_ae
from voxel_autoencoder import build_voxel_ae
import random 
import argparse
from copy import copy 
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import sys
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('--num-cats-excluded', type=int, default=0)
parser.add_argument('--excluded-cats', type=str, default='')
parser.add_argument('--single-cat', type=str, default='')
parser.add_argument('--save-path', type=str)
parser.add_argument('--load-from', type=str, default=None)
parser.add_argument('--concat-features', action='store_true')
parser.add_argument('--no-delta', action='store_false')
parser.add_argument('--num-refine-iters', type=int)
parser.add_argument('--image-code-dim', type=int, default=128)  
parser.add_argument('--vox-code-dim', type=int, default=128)   
parser.add_argument('--batches-per-epoch', type=int, default=100)
parser.add_argument('--num-test-points', type=int, default=1000)
parser.add_argument('--verbosity', type=int, default=0)
parser.add_argument('--num-epochs', type=float, default=100)
parser.add_argument('--final-sigmoid', action='store_true')
parser.add_argument('--tanh-and-add', action='store_true')
parser.add_argument('--average-prior', action='store_true')
parser.add_argument('--k-shot-prior', type=int, default=0)
args = parser.parse_args()
num_refine_iters = args.num_refine_iters
assert (not (args.average_prior and args.k_shot_prior))


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

elif args.single_cat:
    all_categories = [args.single_cat]

def sample_from_multiple_lists(arr_list, num_points):
    indices = random.sample(range(len(arr_list[0])), num_points)
    new_arrs = []
    for arr in arr_list: 
        new_arr = np.array([arr[i] for i in indices])
        new_arrs.append(new_arr)
    return new_arrs


def build_and_train_refiner(load_encoder=True, image_code_dim=512, vox_code_dim=512,
                            num_epochs=1, batches_per_epoch=1, epochs_per_chunk=1,
                            reload_every_chunk=True,
                            regularization=0.0, batch_size=32, vox_encoder_dropout=0.0,
                            image_encoder_dropout=0.0, generator_dropout=0.0, use_batchnorm=False,
                            refiner_path=None, num_test_points=1, average_prior=False,
                            k_shot_prior=0,
                            concat_features=False, voxel_dim=32):
    empty_prior = not (average_prior or k_shot_prior)
    ref_optimizer = optimizers.Adadelta()#lr=0.000001)
    print("Initializing new encoders")
    im_ae = build_image_ae(image_code_dim, regularization=regularization,
                           dropout_rate=image_encoder_dropout, use_batchnorm=use_batchnorm,
                           architecture='large_rgb_conv')
    vox_ae = build_voxel_ae(code_dim=vox_code_dim, regularization=regularization,
                            dropout_rate=vox_encoder_dropout, use_batchnorm=use_batchnorm,
                           voxel_dim=32) 
    im_ae.compile(optimizer=ref_optimizer, loss="binary_crossentropy")
    vox_ae.compile(optimizer=ref_optimizer, loss="binary_crossentropy")

    print("Initializing Refiner")
    vox_ref = delta_refiner.VoxelRefiner(im_ae, vox_ae, regularization=regularization, dropout_rate=generator_dropout,
                                            filter_base_count=32, use_batchnorm=use_batchnorm,
                                          concat_features=concat_features, delta=args.no_delta,
                                          output_dim=voxel_dim, num_im_channels=3,
                                          image_dim=127, final_sigmoid=args.final_sigmoid,
                                          tanh_and_add=args.tanh_and_add)  
    if refiner_path:
        vox_ref.refiner = load_model(refiner_path)
    print(vox_ref.refiner.summary())
    #vox_ref.refiner = multi_gpu_model(vox_ref.refiner, gpus=2)
    print("Loading Train Data")
    print("Compiling Refiner")
    vox_ref.refiner.compile(optimizer=ref_optimizer, loss="binary_crossentropy")
    
    print("Training Refiner")
    train_generator = highres_sampler.triple_generator(category_list=all_categories, n_per_yield=batch_size, mode="train", input_average_voxel=average_prior, k_shot_prior=k_shot_prior)
    
    one_off_train_generator = highres_sampler.triple_generator(category_list=all_categories, n_per_yield=num_test_points, mode='train', input_average_voxel=average_prior, k_shot_prior=k_shot_prior)
    [train_input_images, train_input_voxels], train_target_voxels = next(one_off_train_generator)    
    
    category_to_testing_tuple = {}
    category_to_top_score = {}
    for category in all_categories:
        print("Getting Category %s Testing Data" %category)
        cat_generator = highres_sampler.triple_generator(category_list=[category], n_per_yield=num_test_points, mode="val", 
                                                         input_average_voxel=average_prior, k_shot_prior=k_shot_prior)
        cat_inputs, cat_targets = next(cat_generator)
        category_to_testing_tuple[category] = (cat_inputs, cat_targets)
        category_to_top_score[category] = [0] * num_refine_iters

    top_scores = [0] * num_refine_iters
    top_category_accuracies = [0] * num_refine_iters
    num_epochs_performed = 0
    while num_epochs_performed<num_epochs:
        print("-"*128)
        print("Epoch %d/%.0f" %(num_epochs_performed+1, num_epochs))
        # Get the original data
        # Do the training for each iteration
        if num_refine_iters != 1:
            for batch_i in range(batches_per_epoch):
                [input_images, input_voxels], target_voxels = next(train_generator)
                if empty_prior: input_voxels = np.zeros(input_voxels.shape)
                for iter_i in range(num_refine_iters):
                    vox_ref.refiner.fit([input_images, input_voxels], target_voxels, epochs=1, 
                                        verbose=args.verbosity)
                    input_voxels = vox_ref.refiner.predict([input_images, input_voxels])
                    if args.verbosity and batch_i%100==0:
                        print("%d/%d" %(batch_i, batches_per_epoch))
        else:
             vox_ref.refiner.fit_generator(train_generator, steps_per_epoch=batches_per_epoch, epochs=1, 
                                        verbose=args.verbosity)
        # Now  do testing
        train_scores = []
        copy_train_input_voxels = copy(train_input_voxels)
        for iter_index in range(num_refine_iters):
            score = test_iou4(vox_ref.refiner, train_input_images, copy_train_input_voxels, train_target_voxels)
            train_scores.append(score)
            copy_train_input_voxels = vox_ref.refiner.predict([train_input_images, copy_train_input_voxels])
        print("Train Performance:", train_scores)

        category_current_results = [np.array([])] * len(all_categories)
        for cat_i, category in enumerate(all_categories):
            print("Testing")
            [cat_input_images, cat_input_voxels], cat_target_voxels = category_to_testing_tuple[category]
            if empty_prior: cat_input_voxels = np.zeros(cat_input_voxels.shape)
            print("Test Performance %s:" %category.upper())
            for iter_index in range(num_refine_iters):
                score = test_iou4(vox_ref.refiner, cat_input_images, cat_input_voxels,cat_target_voxels)
                category_current_results[cat_i] = np.append(category_current_results[cat_i], score)
                category_to_top_score[category][iter_index] = max(score, category_to_top_score[category][iter_index])
                cat_input_voxels = vox_ref.refiner.predict([cat_input_images, cat_input_voxels])
            print("\tCurrent Top Scores in %s:" %category.upper(), category_to_top_score[category])
            #print("\tPoint Cloud Method: %.3f, R2N2 1-view: %.3f (%s), R2N2 5-view: %.3f"
            #          %(point_set_results[category],
            #            r2n2_single_results[category],
            #            category_current_results[cat_i].max() - r2n2_single_results[category],
            #            r2n2_5view_results[category]))
        current_category_accuracies = np.average(category_current_results, axis=0)
        for iter_i in range(num_refine_iters):
             top_category_accuracies[iter_i] = max(top_category_accuracies[iter_i], current_category_accuracies[iter_i])
        print("Category-wise Average Performance:", current_category_accuracies, "//Best:", top_category_accuracies)
        num_epochs_performed += 1
        sys.stdout.flush()
        if top_category_accuracies[-1] == current_category_accuracies[-1]:
            vox_ref.refiner.save(args.save_path)
    return vox_ref


def test_iou4(refiner_model, train_images, train_voxels, target_voxels):
    num_test_points = len(train_images)
    preds = refiner_model.predict([train_images, train_voxels])
    threshold = 0.4
    pred_ious = round(iou(target_voxels, preds, threshold=threshold),4)
    #top_score = max(top_score, pred_ious[0]) # was .max but I think there's just 1 element
    score = pred_ious
    # print("Threshold", round(threshold,2), ":", pred_ious)
    return score


def test_refiner(refiner_model, train_images, train_voxels, target_voxels):
    num_test_points = len(train_images)
    preds = refiner_model.predict([train_images, train_voxels])
    top_score = 0
    for t in np.arange(0.1, 0.91, 0.1):
        pred_ious = round(iou(target_voxels, preds, threshold=t),3)
        top_score = max(top_score, pred_ious.max())
        print("Threshold", round(t,2), ":", pred_ious)
    return top_score



def main():
    image_code_dim = args.image_code_dim
    vox_code_dim = args.vox_code_dim
    batch_size = 32
    regularization=0.0 #0.0001 still got test to 0.45 # 0.0005 doesn't get past 0.36 # 0.0003 stuck around 0.42
    image_encoder_dropout = 0.0
    vox_encoder_dropout = 0.0
    generator_dropout = 0.0
    use_batchnorm=False
    num_epochs = args.num_epochs
    batches_per_epoch = args.batches_per_epoch
    epochs_per_chunk = 1
    refiner_path = args.load_from
    num_test_points = args.num_test_points
    average_prior = args.average_prior
    k_shot_prior = args.k_shot_prior
    refiner = build_and_train_refiner(load_encoder=False,
                                      image_code_dim=image_code_dim,
                                      vox_code_dim=vox_code_dim,
                                      batches_per_epoch=batches_per_epoch,
                                      num_epochs=num_epochs,
                                      epochs_per_chunk=epochs_per_chunk,
                                      regularization=regularization,
                                      image_encoder_dropout=image_encoder_dropout,
                                      vox_encoder_dropout=vox_encoder_dropout,
                                      generator_dropout=generator_dropout,
                                      use_batchnorm=use_batchnorm,
                                      batch_size=batch_size,
                                      refiner_path=refiner_path,
                                      num_test_points=num_test_points,
                                      average_prior=average_prior,
                                      k_shot_prior=k_shot_prior,
                                      concat_features=args.concat_features)
    # refiner_model = load_model('vox_ref_%d_%d.h5' %(image_code_dim, vox_code_dim))
    # load_and_test_refiner(image_code_dim, vox_code_dim)


if __name__ == '__main__':
    main()
