from keras.models import load_model
import pickle
from keras import backend as K
from keras.models import Model
import numpy as np
from scipy.special import logit
from iou import iou
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str)
args = parser.parse_args()

gy_categories = ['airplanes',
                 'cars',
                 'chairs',
                 'displays',
                 'phones',
                 'speakers', 
                 'tables']

num_categories = len(gy_categories)
category_to_one_hot = {}
for index, category in enumerate(gy_categories):
        one_hot_vec = np.zeros(num_categories)
        one_hot_vec[index] = 1
        category_to_one_hot[category] = one_hot_vec

model = load_model(args.model_path)
print(model.summary())
vox_encoder = model.layers[3]
for category in gy_categories:
    print(category, vox_encoder.predict(category_to_one_hot[category][np.newaxis,...]))
#print("All 0s", vox_encoder.predict(np.zeros(len(gy_categories))))
#print("All 1s", vox_encoder.predict(np.ones(len(gy_categories))))

