
# coding: utf-8

# In[1]:


import keras

# This is without freezing layers
from highres_sampler import triple_generator
from iou import iou
import pickle
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

learning_rate = 0.2
num_training_iters = 20
num_test_points = 500

def test_iou4(refiner_model, train_images, train_voxels, target_voxels):
    num_test_points = len(train_images)
    preds = refiner_model.predict([train_images, train_voxels])
    threshold = 0.4
    pred_ious = round(iou(target_voxels, preds, threshold=threshold),4)
    #top_score = max(top_score, pred_ious[0]) # was .max but I think there's just 1 element
    score = pred_ious
    # print("Threshold", round(threshold,2), ":", pred_ious)
    return score

def basic_finetuning(model, category, num_shots, lr=0.01, num_val_points=100):
    generator = triple_generator(category_list=[category], n_per_yield=num_shots, mode="train", input_average_voxel=False)
    generator_output = next(generator)
    training_input_batch, training_voxel_batch = generator_output
    
    #generator = triple_generator(category_list=[category], n_per_yield=num_val_points,
    #                             mode="val", input_average_voxel=False)
    #generator_output = next(generator)  
    #val_input_batch, val_voxel_batch = generator_output
    
    generator = triple_generator(category_list=[category], n_per_yield=num_val_points,
                                 mode="test", input_average_voxel=False)
    generator_output = next(generator)  
    test_input_batch, test_voxel_batch = generator_output
    
    
    optimizer = keras.optimizers.SGD(lr=lr)
    model.compile(optimizer=optimizer, loss="binary_crossentropy")
    max_val_score = 0
    counter_without_improvement = 0
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
    for layer in model.layers[2].layers:
        layer.trainable = False
    for layer in model.layers[3].layers:
        layer.trainable = False
    for layer in model.layers[4:-2]:
        layer.trainable=False
    return model


categories = ["benches", "cabinets", "lamps", "rifles", "sofas", "vessels"]
num_shots_arr = [1, 2,3,4,5,10,25]
num_trials = 10

try:
    results_dict = pickle.load(open("finetuning_results_dict_last_layer.pickle", "rb"))
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
            baseline_model = make_model()
            result = basic_finetuning(baseline_model, category, num_shots,
                                      lr=learning_rate, num_val_points=num_test_points)
            performances.append(result)
            del baseline_model
        results_dict[id_tuple] = performances
        print(results_dict)
        pickle.dump(results_dict, open("finetuning_results_dict_last_layer.pickle", "wb"))


