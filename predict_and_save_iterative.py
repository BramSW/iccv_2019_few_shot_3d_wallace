import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
import highres_sampler
import cv2
import sys
import numpy as np
from copy import copy 
from iou import iou

num_points = 200
num_iters = 3
category = "chairs"
generator = highres_sampler.triple_generator(category_list=[category], n_per_yield=num_points, mode="train", k_shot_prior=1, input_average_voxel=False)
[train_im, train_vox], target_vox = next(generator)
refiner = load_model('models/%d_iter.h5' %num_iters)
iteration_voxels = [train_vox]
preds = copy(train_vox)
for _ in range(num_iters):
    preds = refiner.predict([train_im, preds])
    iteration_voxels.append(copy(preds))
    # print(iou(preds, target_vox))
iteration_voxels = np.array(iteration_voxels).squeeze() > 0.4
# print(iou(iteration_voxels[-1], target_vox.squeeze()))

def shift_black_background_to_white(im):
    num_rows, num_cols, _ = im.shape
    for i in range(num_rows):
        for j in range(num_cols):
            if sum(im[i,j,:])==0:
                im[i,j,:] = [1, 1, 1] 
    return im

for i in range(num_points):
    print("Making Figure", i)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    _ = ax.voxels((target_vox[i]>0.4).squeeze(), color='gray', edgecolor='k')
    ax.set_axis_off()
    ax.view_init(160, 200)
    plt.savefig('iterative_examples/%s/ground_truth_%d.png' %(category, i))
    plt.close()

    fig = plt.figure()
    #fig, axes = plt.subplots((1,num_iters+1))
    for iter_i in range(num_iters+1):
        ax = fig.add_subplot(1, num_iters+2, iter_i+1, projection='3d')
        iter_voxel = iteration_voxels[iter_i][i]
        _ = ax.voxels(iter_voxel, color="blue", edgecolor='k')
        ax.set_axis_off()
        ax.view_init(160, 200)
        np.save("iterative_examples/%s/pred%d_%d.npy" %(category, iter_i,i), iter_voxel)
    ax = fig.add_subplot(1, num_iters+2, num_iters+2, projection='3d')
    _ = ax.voxels(target_vox[i].squeeze(), color='blue', edgecolor='k')
    ax.set_axis_off()
    ax.view_init(160, 200)
    # print(iteration_voxels[num_iters][i].shape, target_vox[i].shape)
    score = round(iou(iteration_voxels[num_iters][i], target_vox[i].squeeze()), 3) * 1000
    # print("score", score)
    plt.savefig('iterative_examples/%s/pred_%d_%d.png' %(category, i, score))
    
    train_im[i] = shift_black_background_to_white(train_im[i])
    cv2.imwrite('iterative_examples/%s/image_%d.png' %(category,i), 255*train_im[i])
    plt.close()
    np.save("iterative_examples/%s/truth_%d.npy" %(category,i), (target_vox[i]).squeeze())
