import keras
import pickle
from iou import iou
import numpy as np

old_model = keras.models.load_model('gy_categories_vox_dim_1.h5')
im_input = keras.layers.Input(shape=(32,32,1))
code_input = keras.layers.Input(shape=(1,))
image_encoder = old_model.layers[2]
im_code = image_encoder(im_input)
x = keras.layers.concatenate([im_code, code_input])
for layer in old_model.layers[5:13]:
    x = layer(x)
prior_input = keras.layers.Input(shape=(32,32,32,1))
x = keras.layers.add([x, prior_input])
x = keras.layers.Activation('sigmoid')(x)
new_model = keras.Model(inputs=[im_input, code_input, prior_input], outputs=x)
new_model.summary()

input_voxels = pickle.load(open('input_voxels.pickle', 'rb'))
input_images = pickle.load(open('input_images.pickle', 'rb'))
target_voxels  = pickle.load(open('target_voxels.pickle', 'rb'))
test_category = "rifles"
# data_index = 3

num_training_points = 5
num_testing_points = 50
min_test_data_index = 50
assert min_test_data_index >= num_training_points

test_data_index_list = range(min_test_data_index, min_test_data_index + num_testing_points)
num_codes = 20
code_performances = np.zeros((num_codes, num_data_points))
baselines = []


for data_index in data_index_list:
    input_image = input_images[test_category][data_index][np.newaxis, ...]
    target_voxel = target_voxels[test_category][data_index][np.newaxis, ...]
    prior_voxel = input_voxels[test_category][data_index_list][np.newaxis, ...]
    baseline_output = old_model.predict([input_image, prior_voxel])
    baseline_score = iou(baseline_output, target_voxel)
    print("Baseline:", baseline_score)
    baselines.append(baseline_score)
    for code in range(num_codes):
        new_output = new_model.predict([input_image, np.array([code]), prior_voxel])
        score = iou(new_output, target_voxel)
        print(code, score)
        code_performances[code][data_index - min_data_index] = score
print("Average Baseline:", np.average(baselines))
print("Average Code Performance:", np.average(code_performances, axis=1))


"""
Codes for models/concat_vox_dim_1
airplanes [[6.6943183]]
benches [[15.674586]]
cabinets [[62.523083]]
cars [[0.]]
chairs [[0.]]
displays [[41.311626]]
lamps [[6.5434856]]
phones [[33.514797]]
rifles [[3.7708092]]
sofas [[10.934689]]
speakers [[21.81574]]
tables [[29.042206]]
vessels [[2.0796802]]
All 0s [[25.483665]]
All 1s [[16.547955]]
"""
