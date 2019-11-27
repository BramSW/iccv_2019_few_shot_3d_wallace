import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sampler
import cv2
import sys
import numpy as np
import binvox_rw
import os 
from pprint import pprint

base_path = "/data/bw462/3d_recon/ShapeNetCore/simple_category_sorted/"
categories = os.listdir(base_path)
category_to_binvox_paths = {cat:[base_path + cat + '/' + file_name for file_name in os.listdir(base_path + cat)]
                            for cat in categories}

for category in category_to_binvox_paths:
    for binvox_path in category_to_binvox_paths[category][:2]:
        index = 0
        print(binvox_path)
        binvox_id = binvox_path.split('/')[-1].split('.')[0]
        # binvox_path = base_path + "../ShapeNetCore.v2/" + category + "/" + binvox_id + "/models/model_normalized.surface.binvox"  ###
        print("\tReading")
        vox_array = binvox_rw.read_as_3d_array(open(binvox_path, "rb")).data#[::2,::2,::2]###
        print(vox_array.shape)
        x, y, z = np.where(vox_array)
        image_dir = base_path + category + '/' + binvox_id + '_images/'
        print("\tMaking Directory")
        os.mkdir(image_dir)
        print("\tDrawing")
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.axis('off')
        _ = ax.voxels(vox_array.squeeze(), color='gray')
        print("\tSaving")
        plt.savefig(image_dir + 'image_%d.png' %index)
        plt.close()
        index += 1

