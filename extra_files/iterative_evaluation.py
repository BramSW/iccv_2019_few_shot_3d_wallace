import numpy as np
from keras import models
import sampler
from iou import iou
import matplotlib.pyplot as plt
from pprint import pprint
import pickle
import argparse
import tensorflow as tf
import random
from copy import copy
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('--use-prior', action='store_true')
parser.add_argument('--category-agnostic', action='store_true')
parser.add_argument('--num-iters', type=int, default=10)
parser.add_argument('--one-shot-prior', action='store_true')
parser.add_argument('--model-path', type=str)
parser.add_argument('--bootstrap-path', type=str, default='')
parser.add_argument('--gy-cats', action='store_true')
args = parser.parse_args()


if args.gy_cats:
    all_categories = ["airplanes",
                      "cars",
                      "chairs",
                      "displays",
                      "phones",
                      "speakers",
                      "tables"]
else:
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


def iteratively_test_model(model_path, num_iters=3, threshold=0.4, num_test_points=1000, empty_prior=True,
                           create_test_data=False, load_input_images=True, 
                           load_target_voxels=True, one_shot_prior=False):
    assert (not (empty_prior and one_shot_prior))
    if create_test_data:
        raise NotImplementedError
    else:
        if load_input_images:
            category_to_input_images = pickle.load(open('val_input_images.pickle', 'rb'))
        else:
            raise NotImplementedError
        if load_target_voxels:
            category_to_target_voxels = pickle.load(open('val_target_voxels.pickle', 'rb'))
        else:
            raise NotImplementedError
        if not empty_prior:         
            if args.category_agnostic: 
                print("Category Agnostic Prior")
                category_to_input_voxels = pickle.load(open('overall_training_average_voxels.pickle', 'rb'))
            elif args.bootstrap_path: 
                print("Bootstrapping prior")
                image_only_model = models.load_model(args.bootstrap_path)
                category_to_input_voxels = {}
                for category in all_categories:
                    category_to_input_voxels[category] = image_only_model.predict([category_to_input_images[category],
                                                                                  np.zeros(category_to_target_voxels[category].shape)])
            elif one_shot_prior:
                print("Using 1-shot prior")
                category_to_input_voxels = pickle.load(open('train_target_voxels.pickle', 'rb'))
                for category in all_categories:
                    random.shuffle(category_to_input_voxels[category])
            else:
                print("Using average prior")
                category_to_input_voxels = pickle.load(open('train_input_voxels.pickle', 'rb'))
        else:
            print("Using empty prior")
            #category_to_input_voxels = pickle.load(open('target_voxels.pickle', 'rb'))
            category_to_input_voxels = {category:np.zeros(category_to_target_voxels[category].shape)
                                        for category in all_categories}
    category_to_score_arr = {cat:[] for cat in all_categories}
    model = models.load_model(model_path)
    print(model.summary())
    for i in range(num_iters):
        for category in all_categories:
            preds = model.predict([category_to_input_images[category], category_to_input_voxels[category]])
            iou_score = iou(category_to_target_voxels[category], preds, threshold=threshold)
            category_to_score_arr[category].append(round(iou_score,3))
            category_to_input_voxels[category] = preds
    return category_to_score_arr


def main():
    display_images = False
    num_iters=10
    threshold_to_category_to_score_arr = {}
    thresholds = [0.4] #np.arange(0.3, 0.61, 0.01)

    for threshold in thresholds:
        threshold_to_category_to_score_arr[threshold] = iteratively_test_model(args.model_path,
                                                                               num_iters=args.num_iters,
                                                                               num_test_points=100,
                                                                               threshold=threshold,
                                                                               empty_prior=not args.use_prior,
                                                                               create_test_data=False,
                                                                               one_shot_prior=args.one_shot_prior)
    pprint(threshold_to_category_to_score_arr)
    # Currently just printing average of last one
    pprint(np.average(np.array(list(threshold_to_category_to_score_arr[threshold].values())), axis=0))
    if display_images:
        for category in all_categories:
            plt.figure()
            for threshold in thresholds:
                score_list = threshold_to_category_to_score_arr[threshold][category]
                plt.plot(score_list, label="Threshold=%0.2f" %threshold)
            plt.title(category)
            plt.show()
            plt.close()


if __name__ == '__main__':
    main()

