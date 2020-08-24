import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
import highres_sampler
import cv2
import sys
import numpy as np

num_points = 20

generator = highres_sampler.triple_generator(category_list=['lamps'], n_per_yield=num_points, mode="train")
[train_im, train_vox], target_vox = next(generator)
refiner = load_model('models/gy_cats_2_iter_NONdelta.h5')
direct = load_model("models/gy_cats_RGB_baseline.h5")


refined_vox = refiner.predict([train_im, train_vox])
refined_vox = refiner.predict([train_im, refined_vox])

from_image_vox = direct.predict([train_im, np.zeros(train_vox.shape)])

for i in range(num_points):
    print("Making Figure", i)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    _ = ax.voxels((target_vox[i]>0.5).squeeze(), color='gray', edgecolor='k')
    ax.set_axis_off()
    ax.view_init(-65, 85)
    plt.savefig('ground_truth_%d.png' %i)
    plt.close()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    _ = ax.voxels((refined_vox[i]>0.4).squeeze(), color="blue", edgecolor='k')
    ax.set_axis_off()
    ax.view_init(-65, 85)
    plt.savefig('pred_%d.png' %i)
    plt.close()


    cv2.imwrite('image_%d.png' %i, 255*train_im[i])
    plt.close()
