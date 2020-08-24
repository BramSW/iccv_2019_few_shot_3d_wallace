from keras.models import load_model
from refiner import VoxelRefiner
from binvox_rw import read_as_3d_array
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import pickle
from iou import iou
import highres_sampler
from image_autoencoder import build_image_ae
from voxel_autoencoder import build_voxel_ae

all_categories = [#"airplanes",
                      "benches",
                      "cabinets",
                      #"cars",
                      #"chairs",
                      #"displays",
                      "lamps",
                      #"phones",
                      "rifles",
                      "sofas",
                      #"speakers",
                      #"tables",
                      "vessels"]

num_test_points = 100
category_to_input_voxels = {}
category_to_target_voxels = {}
for num_shots in [1, 2, 3, 4, 5, 10, 25]:
    for category in all_categories:
        gen = highres_sampler.triple_generator(category_list=[category], n_per_yield=num_test_points,
                                           k_shot_prior=num_shots, input_average_voxel=False, mode="test")
        [_, input_voxels], target_voxels = next(gen)
        category_to_input_voxels[category] = input_voxels
        category_to_target_voxels[category] = target_voxels

    # print("Threshold:", round(t,2))
    threshold_scores = []
    for category in all_categories:
        #print(category, ":", end="")
        input_voxels = category_to_input_voxels[category]
        target_voxels = category_to_target_voxels[category]
        score =  round(iou(target_voxels, input_voxels, threshold=0.4),3)
        #print("\tGuess Fed Voxel:\t", score)
        threshold_scores.append(score)
    print(num_shots, np.average(threshold_scores))
    for cat, score in zip(all_categories, threshold_scores): print(cat, score)
    print()
