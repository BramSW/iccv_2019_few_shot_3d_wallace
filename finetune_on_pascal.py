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
parser.add_argument('--prior', action='store_true')
parser.add_argument('--num-cats-excluded', type=int, default=0)
parser.add_argument('--excluded-cats', type=str, default='')
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
parser.add_argument('--final-sigmoid', action='store_true')
parser.add_argument('--tanh-and-add', action='store_true')
parser.add_argument('--average-prior', action='store_true')
parser.add_argument('--k-shot-prior', type=int, default=0)
args = parser.parse_args()
num_refine_iters = args.num_refine_iters
prior = args.prior

assert (not (args.average_prior and args.k_shot_prior))


def build_and_train_refiner(load_path, load_encoder=True, image_code_dim=512, vox_code_dim=512,
                            num_epochs=1, batches_per_epoch=1, epochs_per_chunk=1,
                            reload_every_chunk=True,
                            regularization=0.0, batch_size=32, vox_encoder_dropout=0.0,
                            image_encoder_dropout=0.0, generator_dropout=0.0, use_batchnorm=False,
                            num_test_points=1, average_prior=False,
                            k_shot_prior=0,
                            concat_features=False, voxel_dim=32):
    empty_prior = not (average_prior or k_shot_prior)
    ref_optimizer = optimizers.Adadelta(lr=0.1)#lr=0.000001)
    # ref_optimizer = optimizers.Adam()
    # ref_optimizer = optimizers.SGD(lr=1e-4)
    print("Initializing new encoders")
    im_ae = build_image_ae(image_code_dim, regularization=regularization,
                           dropout_rate=image_encoder_dropout, use_batchnorm=use_batchnorm,
                           architecture='large_rgb_conv')
    vox_ae = build_voxel_ae(code_dim=vox_code_dim, regularization=regularization,
                            dropout_rate=vox_encoder_dropout, use_batchnorm=use_batchnorm,
                           voxel_dim=32)
    vox_ref = delta_refiner.VoxelRefiner(im_ae, vox_ae, regularization=regularization, dropout_rate=generator_dropout,
                                            filter_base_count=32, use_batchnorm=use_batchnorm,
                                          concat_features=concat_features, delta=args.no_delta,
                                          output_dim=voxel_dim, num_im_channels=3,
                                          image_dim=127, final_sigmoid=args.final_sigmoid,
                                          tanh_and_add=args.tanh_and_add)  
    vox_ref.refiner = load_model(load_path)
    print(vox_ref.refiner.summary())
    #vox_ref.refiner = multi_gpu_model(vox_ref.refiner, gpus=2)
    print("Loading Train Data")
    print("Compiling Refiner")
    vox_ref.refiner.compile(optimizer=ref_optimizer, loss="binary_crossentropy")
    train_generator = highres_sampler.pascal_sampler('shapenet', n_per_yield=batch_size, mode="train", prior=prior)
    train_inputs, train_target_voxels = next(train_generator)
    val_generator = highres_sampler.pascal_sampler('shapenet', n_per_yield=497, mode="test", prior=prior, balance_classes=True)
    val_inputs, val_target_voxels = next(val_generator)
    novel_generator = highres_sampler.pascal_sampler('novel', n_per_yield=501, mode="train", prior=prior, balance_classes=True)
    novel_inputs, novel_target_voxels = next(novel_generator)
    
    print("Training Refiner")
    """
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
    """
    num_epochs_performed = 0
    print("Train", test_iou4(vox_ref.refiner, train_inputs[0], train_inputs[1], train_target_voxels))
    print("Identity Guess", iou(train_inputs[1], train_target_voxels, 0.4))
    print("Val", test_iou4(vox_ref.refiner, val_inputs[0], val_inputs[1], val_target_voxels))
    print("Novel", test_iou4(vox_ref.refiner, novel_inputs[0], novel_inputs[1], novel_target_voxels))
    print("Identity Guess", iou(novel_inputs[1], novel_target_voxels, 0.4))
    while num_epochs_performed<num_epochs:
        print("-"*128)
        print("Epoch %d/%.0f" %(num_epochs_performed+1, num_epochs))
        vox_ref.refiner.fit_generator(train_generator, steps_per_epoch=batches_per_epoch, epochs=1, 
                                    verbose=args.verbosity)
        if num_epochs_performed % 1 ==0:
            print("Train", test_iou4(vox_ref.refiner, train_inputs[0], train_inputs[1], train_target_voxels))
            print("Identity Guess", iou(train_inputs[1], train_target_voxels, 0.4))
            print("Val", test_iou4(vox_ref.refiner, val_inputs[0], val_inputs[1], val_target_voxels))
            print("Novel", test_iou4(vox_ref.refiner, novel_inputs[0], novel_inputs[1], novel_target_voxels))
            print("Identity Guess", iou(novel_inputs[1], novel_target_voxels, 0.4))
            # print("Val", vox_ref.refiner.evaluate(val_inputs, val_target_voxels))
            # print("Novel", vox_ref.refiner.evaluate(novel_inputs, novel_target_voxels))
        num_epochs_performed  += 1
        """
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
        """
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
    batch_size = 64
    regularization=0.0 #0.0001 still got test to 0.45 # 0.0005 doesn't get past 0.36 # 0.0003 stuck around 0.42
    image_encoder_dropout = 0.0
    vox_encoder_dropout = 0.0
    generator_dropout = 0.0
    use_batchnorm=False
    num_epochs = 500
    batches_per_epoch = args.batches_per_epoch
    epochs_per_chunk = 1
    load_path = args.load_from
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
                                      load_path=load_path,
                                      num_test_points=num_test_points,
                                      average_prior=average_prior,
                                      k_shot_prior=k_shot_prior,
                                      concat_features=args.concat_features)
    # refiner_model = load_model('vox_ref_%d_%d.h5' %(image_code_dim, vox_code_dim))
    # load_and_test_refiner(image_code_dim, vox_code_dim)


if __name__ == '__main__':
    main()
