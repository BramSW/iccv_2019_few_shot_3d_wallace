from binvox_rw import read_as_3d_array
import numpy as np
import os
import pickle
import random
import cv2
from sklearn.model_selection import train_test_split
from pprint import pprint 
import shutil

base_path = "/data/bw462/3d_recon/ShapeNetCore/ShapeNetCore.v2/"
all_categories = os.listdir(base_path)
all_categories.remove('taxonomy.json')
category_to_object_list = {cat:os.listdir(base_path + cat)
                           for cat in all_categories}
# print(category_to_object_list.items())
category_to_voxel_paths = {cat:[base_path + cat + '/' + obj + '/models/model_normalized.solid.binvox'
                           for obj in obj_list]
                           for cat, obj_list in category_to_object_list.items()}
"""
num_in_category = np.array([len(arr) for arr in category_to_object_list.values()])
print(category_to_object_list.keys())
print(category_to_object_list['04090263'])
print(len(category_to_object_list['04090263']))
print(num_in_category)
print(sum(num_in_category))
print(num_in_category.min())
pprint(category_to_voxel_paths)
"""

new_base = "/data/bw462/3d_recon/ShapeNetCore/simple_category_sorted/"
os.mkdir(new_base)
for category in all_categories:
    os.mkdir(new_base + category)
    for voxel_path in category_to_voxel_paths[category]:
        obj_id = voxel_path.split('/')[-3]
        try:
            shutil.copyfile(voxel_path, new_base + category + '/' + obj_id + ".binvox")
        except FileNotFoundError:
            pass
