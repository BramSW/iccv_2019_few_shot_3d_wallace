import numpy as np
from keras import models
import iou
import matplotlib.pyplot as plt
from pprint import pprint
import pickle
import argparse
import tensorflow as tf
import random
from copy import copy
from keras.backend.tensorflow_backend import set_session
import highres_sampler
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('--num-iters', type=int, default=10)
parser.add_argument('--average-prior', action='store_true')
parser.add_argument('--k-shot-prior', type=int, default=0)
parser.add_argument('--gy-cats', action='store_true')
parser.add_argument('--num-test-points', type=int, default=100)
parser.add_argument('--threshold', type=float, default=0.4)
parser.add_argument('--transfer-cats', action='store_true')
parser.add_argument('--randomize-input-cats', action='store_true')
parser.add_argument('--model-path', type=str)
args = parser.parse_args()

assert (not (args.gy_cats and args.transfer_cats))

if args.gy_cats:
    all_categories = ["airplanes",
                      "cars",
                      "chairs",
                      "displays",
                      "phones",
                      "speakers",
                      "tables"]
elif args.transfer_cats:
    all_categories = ["benches",
                      "cabinets",
                      "vessels", 
                      "lamps",
                      "rifles",
                      "sofas"]

def iteratively_test_model(model_path, num_iters=3, threshold=0.4, num_test_points=1000, average_prior=False,
                           create_test_data=False, load_input_images=True, 
                           load_target_voxels=True, k_shot_prior=0, randomize_categories=False):
    assert (not (average_prior and k_shot_prior))
    num_batches = num_test_points // 100
    category_to_score_arr = {cat:[] for cat in all_categories}
    model = models.load_model(model_path)
    # print(model.summary())
    for category in all_categories:
        print(category)
        # generator = highres_sampler.triple_generator(category_list=[category], n_per_yield=100, mode="val", input_average_voxel = average_prior,
        #                                              k_shot_prior=k_shot_prior, randomize_categories=randomize_categories)
        generator = highres_sampler.full_test_generator(category_list=[category], n_per_yield=100, mode="test", input_average_voxel = average_prior,
                                                        k_shot_prior=k_shot_prior, randomize_categories=randomize_categories)
        all_iter_ious = [ [] for _ in range(num_iters)]
        for [input_im, input_vox], target_vox in generator:
            input_vox = target_vox
            for iter_i in range(num_iters):
                preds = model.predict([input_im, input_vox])
                iou_scores = iou.iou_list(preds, target_vox, threshold=threshold)
                input_vox = preds
                all_iter_ious[iter_i].extend(iou_scores)
        category_to_score_arr[category] = all_iter_ious
    return category_to_score_arr


def main():
    display_images = False
    num_iters=10
    threshold_to_category_to_score_arr = {}
    thresholds = [args.threshold] #np.arange(0.3, 0.61, 0.01)

    for threshold in thresholds:
        category_to_score_arr = iteratively_test_model(args.model_path,
                                                                               num_iters=args.num_iters,
                                                                               num_test_points=args.num_test_points,
                                                                               threshold=threshold,
                                                                               average_prior=args.average_prior,
                                                                               k_shot_prior=args.k_shot_prior,
                                                                               randomize_categories=args.randomize_input_cats)
    # pprint(threshold_to_category_to_score_arr)
    # Currently just printing average of last one
    # pprint(np.average(np.array(list(threshold_to_category_to_score_arr[threshold].values())), axis=0))
    pickle_save_string = ("test_performances/transfer_" if args.transfer_cats else "test_performances/gy_") + "cats_perfect_prior_testing_peformance_" + args.model_path.split('/')[-1].split('.')[0] + ".pickle"
    pickle.dump(category_to_score_arr, open(pickle_save_string, 'wb'))

if __name__ == '__main__':
    main()

