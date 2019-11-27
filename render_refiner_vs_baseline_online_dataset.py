import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
import pickle
import highres_sampler
import cv2
import sys
import numpy as np
import random
import os

category_name_to_id = {"airplanes":"02691156",
                       "benches":"02828884",
                       "cabinets":"02933112",
                       "cars":"02958343",
                       "chairs":"03001627",
                       "displays":"03211117",
                       "lamps":"03636649",
                       "phones":"04401088",
                       "rifles":"04090263",
                       "sofas":"04256520",
                       "speakers":"03691459",
                       "tables":"04379243",
                       "vessels":"04530566"
                       }


def online_dataset_sample(category, n_per_yield=5):
    category_id = category_name_to_id[category]
    prior_voxel = pickle.load(open('/data/bw462/3d_recon/category_to_average_train_voxel.pickle', 'rb'))[category_id]
    input_voxels = np.array([prior_voxel for _ in range(n_per_yield)])
    im_dir = '/data/bw462/Stanford_Online_Products/{}_final/'.format(category[:-1])
    im_paths = [im_dir + filename for filename in random.choices(os.listdir(im_dir), k=n_per_yield)]
    ims = np.array([cv2.resize(cv2.imread(im_path), (127, 127)) / 255 for im_path in im_paths])
    return ims, input_voxels


# Mainly need to make new generator?
category ='lamps'
n = 200
train_im, train_vox = online_dataset_sample(category, n_per_yield=n)
refiner = load_model('/data/bw462/3d_recon/models/1_iter_1_shot.h5')
direct = load_model("/data/bw462/3d_recon/models/baseline.h5")


refined_vox = refiner.predict([train_im, train_vox])

from_image_vox = direct.predict([train_im, np.zeros(train_vox.shape)])

for i in range(n):
    print("Making Figure", i)
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_title('Ours')
    _ = ax.voxels((from_image_vox[i]>0.5).squeeze(), color='gray', edgecolor='k')
    ax.set_axis_off()
    ax.view_init(-65, 85)
    # plt.savefig('baseline_%d.png' %i)
    # plt.close()

    # fig = plt.figure()
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.set_title('Baseline')
    _ = ax.voxels((refined_vox[i]>0.4).squeeze(), color="blue", edgecolor='k')
    ax.set_axis_off()
    ax.view_init(-65, 85)
    # plt.savefig('pred_%d.png' %i)
    # plt.close()

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(train_im[i])

    plt.axis('off')
    plt.savefig('online/{}/result_{}.png'.format(category, i))
    plt.close()
