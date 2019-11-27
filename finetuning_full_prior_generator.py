
# coding: utf-8

# In[1]:


import keras

# This is without freezing layers
from highres_sampler import triple_generator
from iou import iou
import pickle
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
#set_session(sess)

num_batches = 200
lr = 0.005
num_test_points = 1000
validate = True
num_trials = 1

def test_iou4(refiner_model, train_images, train_voxels, target_voxels):
    num_test_points = len(train_images)
    preds = refiner_model.predict([train_images, train_voxels])
    threshold = 0.4
    pred_ious = round(iou(target_voxels, preds, threshold=threshold),4)
    #top_score = max(top_score, pred_ious[0]) # was .max but I think there's just 1 element
    score = pred_ious
    # print("Threshold", round(threshold,2), ":", pred_ious)
    return score


def make_model():
    model = keras.models.load_model('/data/bw462/3d_recon/models/baseline.h5')
    for layer in model.layers[2].layers:
        layer.trainable = False
    for layer in model.layers[3].layers:
        layer.trainable = False
    return model

def basic_finetuning(model, category, lr=0.01, num_val_points=100):
    training_generator = triple_generator(category_list=[category], n_per_yield=32, mode="train", input_average_voxel=False)
    generator_output = next(training_generator)
    training_input_batch, training_voxel_batch = generator_output
    
    if validate:
        generator = triple_generator(category_list=[category], n_per_yield=num_val_points,
                                     mode="val", input_average_voxel=False)
        generator_output = next(generator)  
        val_input_batch, val_voxel_batch = generator_output
    
    generator = triple_generator(category_list=[category], n_per_yield=num_val_points,
                                 mode="test", input_average_voxel=False)
    generator_output = next(generator)  
    test_input_batch, test_voxel_batch = generator_output
    
    
    optimizer = keras.optimizers.SGD(lr=lr)
    model.compile(optimizer=optimizer, loss="binary_crossentropy")
    max_val_score = 0
    counter_without_improvement = 0
    for batch_i in range(10):
        print(batch_i)
        model.fit_generator(training_generator, steps_per_epoch=100, verbose=1)
        if validate: 
            train_score = test_iou4(model, training_input_batch[0], training_input_batch[1], training_voxel_batch)
            print("train:", train_score)
            current_val_score = test_iou4(model, val_input_batch[0], val_input_batch[1], val_voxel_batch)
            print("val:", current_val_score)
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

categories = ["benches", "cabinets", "lamps", "rifles", "sofas", "vessels"]

try:
    results_dict = pickle.load(open("finetuning_full_prior_generator_results_dict.pickle", "rb"))
except:
    results_dict = {}
for category in categories:
        if category in results_dict: continue
        print(category)
        performances = []
        for trial_i in range(num_trials):
            print(trial_i)
            baseline_model = make_model()
            result = basic_finetuning(baseline_model, category,
                                      lr=lr, num_val_points=num_test_points)
            performances.append(result)
            del baseline_model
        results_dict[category] = performances
        print(results_dict)
        pickle.dump(results_dict, open("finetuning_full_prior_generator_results_dict.pickle", "wb"))


