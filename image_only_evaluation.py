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
parser.add_argument('--model-path', type=str)
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


def iteratively_test_model(model_path, threshold=0.4):
    print("Loading Data")
    category_to_input_images = pickle.load(open('val_input_images.pickle', 'rb'))
    category_to_target_voxels = pickle.load(open('val_target_voxels.pickle', 'rb'))
       
    category_to_score_arr = {cat:[] for cat in all_categories}
    print("Loading Model")
    model = models.load_model(model_path)
    print("Testing")
    for category in all_categories:
        print("\t", category)
        preds = model.predict(category_to_input_images[category])
        iou_score = iou(category_to_target_voxels[category], preds, threshold=threshold)
        category_to_score_arr[category] = round(iou_score,3)
    return category_to_score_arr


def main():
    scores = iteratively_test_model(args.model_path)
    pprint(scores)
    pprint(np.average(list(scores.values())))

if __name__=='__main__':
    main()
