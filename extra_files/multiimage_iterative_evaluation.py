import numpy as np
from keras import models
import sampler
from iou import iou
import matplotlib.pyplot as plt
from pprint import pprint
import pickle
import argparse
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('--use-prior', action='store_true')
parser.add_argument('--category-agnostic', action='store_true')
parser.add_argument('--create-data', action='store_true')
parser.add_argument('--num-iters', type=int, default=10)
parser.add_argument('--model-path', type=str)
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
                      "vessels"]


def iteratively_test_model(model_path, num_iters=3, threshold=0.4, num_test_points=1000, empty_prior=True,
                           create_test_data=True):
    if create_test_data:
        multiimage_samples = sampler.get_n_multiimage_samples(n=num_test_points,
                                                     empty_prior=empty_prior,
                                                     mode="val")
        pickle.dump(multiimage_samples, open('multiimage_samples.pickle', 'wb'))
    else:
        multiimage_samples = pickle.load(open('multiimage_samples.pickle', 'rb'))
    input_image_arrs, input_voxels, target_voxels = multiimage_samples

    category_to_score_arr = {cat:[] for cat in all_categories}
    model = models.load_model(model_path)
    for category in all_categories:
        if empty_prior:
            cat_input_voxels = np.zeros((num_test_points, 32, 32, 32, 1))
        else:
            cat_input_voxels = np.array(input_voxels[category])
        for i in range(num_iters):
            images = np.array([image_arr[i] for image_arr in input_image_arrs[category]])[...,np.newaxis]
            preds = model.predict([images, cat_input_voxels])
            iou_score = iou(np.array(target_voxels[category])[...,np.newaxis], preds, threshold=threshold)
            category_to_score_arr[category].append(iou_score)
            cat_input_voxels = preds
    return category_to_score_arr


def main():
    display_images = False
    num_iters=10
    threshold_to_category_to_score_arr = {}
    thresholds = [0.4] #np.arange(0.3, 0.61, 0.01)
    num_test_points = 20
    for threshold in thresholds:
        threshold_to_category_to_score_arr[threshold] = iteratively_test_model(args.model_path,
                                                                               num_iters=args.num_iters,
                                                                               num_test_points=num_test_points,
                                                                               threshold=threshold,
                                                                               empty_prior=not args.use_prior,
                                                                               create_test_data=args.create_data)
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

