
# coding: utf-8

# In[1]:


import keras

# This is without freezing layers
from highres_sampler import multiview_triple_generator
from iou import iou
import pickle
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np
import sys

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

learning_rate = 0.005
num_training_iters = 200
num_test_points = 100

def test_iou4(refiner_model, train_images, train_voxels, target_voxels):
    num_test_points = len(train_images)
    preds = refiner_model.predict([train_images, train_voxels])
    threshold = 0.4
    pred_ious = round(iou(target_voxels, preds, threshold=threshold),4)
    #top_score = max(top_score, pred_ious[0]) # was .max but I think there's just 1 element
    score = pred_ious
    # print("Threshold", round(threshold,2), ":", pred_ious)
    return score

def split_input_batch_image_list(input_batch, target_batch, images_per_model):
    image_lists, voxels = input_batch
    num_models = image_lists.shape[0]
    new_image_list = []
    new_voxel_list = []
    new_target_list = []

    for model_i in range(num_models):
        for image_i in range(images_per_model):
            new_image_list.append(image_lists[model_i, image_i])
            new_voxel_list.append(voxels[model_i])
            new_target_list.append(target_batch[model_i])
    new_image_list = np.array(new_image_list)
    new_voxel_list = np.array(new_voxel_list)
    new_target_list = np.array(new_target_list)
    print(new_image_list.shape, new_voxel_list.shape, new_target_list.shape)
    return [new_image_list, new_voxel_list], new_target_list

    


def basic_finetuning(model, category, num_shots, images_per_model, lr=0.01, num_val_points=100):
    print("\tGetting data")
    generator = multiview_triple_generator(category_list=[category], n_per_yield=num_shots, mode="train", input_average_voxel=False)
    generator_output = next(generator)
    training_input_batch, training_voxel_batch = generator_output
    
    
    generator = multiview_triple_generator(category_list=[category], n_per_yield=num_val_points,
                                 mode="test", input_average_voxel=False)
    generator_output = next(generator)
    test_input_batch, test_voxel_batch = generator_output
    del generator
    # Right now training_input_batch and test_input_batch have an image list instead of single images
    # Want to break these out into separate datapoints with repeated associated voxels
    training_input_batch, training_voxel_batch = split_input_batch_image_list(training_input_batch, training_voxel_batch, images_per_model)
    test_input_batch, test_voxel_batch = split_input_batch_image_list(test_input_batch, test_voxel_batch, images_per_model)


    optimizer = keras.optimizers.SGD(lr=lr)
    model.compile(optimizer=optimizer, loss="binary_crossentropy")
    max_val_score = 0
    counter_without_improvement = 0
    print("\tFitting")
    for batch_i in range(num_training_iters):
        #print(batch_i)
        model.fit(training_input_batch, training_voxel_batch, verbose=0)
        #train_score = test_iou4(model, training_input_batch[0], training_input_batch[1], training_voxel_batch)
        #print("train:", train_score)
        #current_val_score = test_iou4(model, val_input_batch[0], val_input_batch[1], val_voxel_batch)
        #print("val:", current_val_score)
        #if current_val_score > max_val_score:
        #    counter_without_improvement = 0
        #    max_val_score = current_val_score
        #else: # current is equal or worse
        #    counter_without_improvement += 1
        #    if counter_without_improvement > max_stable_iters:
        #        break
        #max_val_score = max(max_val_score, current_val_score)
        #print()
    return test_iou4(model, test_input_batch[0], test_input_batch[1], test_voxel_batch)


def make_model():
    model = keras.models.load_model('/data/bw462/3d_recon/models/baseline.h5')
    return model


categories = ["benches", "cabinets", "lamps", "rifles", "sofas", "vessels"]
num_shots_arr = [1, 2, 3, 4, 5, 10, 25]
num_trials = 10
images_per_model = 5

try:
    results_dict = pickle.load(open("finetuning_results_dict_whole_model.pickle", "rb"))
except:
    results_dict = {}
for category in categories:
    for num_shots in num_shots_arr:
        id_tuple = (category, num_shots)
        print(id_tuple)
        if id_tuple in results_dict: continue
        performances = []
        for trial_i in range(num_trials):
            print(trial_i)
            print("Making model")
            baseline_model = make_model()
            print("Finetuning")
            result = basic_finetuning(baseline_model, category, num_shots, images_per_model,
                                      lr=learning_rate, num_val_points=num_test_points)
            performances.append(result)
            del baseline_model
        results_dict[id_tuple] = performances
        print(results_dict)
        sys.stdout.flush()
        pickle.dump(results_dict, open("finetuning_results_dict_whole_model.pickle", "wb"))


