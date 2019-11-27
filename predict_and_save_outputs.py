import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
import highres_sampler
import cv2
import sys
import numpy as np

def shift_black_background_to_white(im):
    num_rows, num_cols, _ = im.shape
    for i in range(num_rows):
        for j in range(num_cols):
            if sum(im[i,j,:])==0:
                im[i,j,:] = [1, 1, 1] 
    return im

num_points = 10

category_list = ['benches', 'cabinets', 'rifles', 'vessels', 'sofas', 'lamps']
refiner = load_model('/data/bw462/3d_recon/models/1_iter_1_shot.h5')
baseline = load_model('/data/bw462/3d_recon/models/baseline.h5')

for category in category_list:
    generator = highres_sampler.triple_generator(category_list=[category], n_per_yield=num_points, mode="test", input_average_voxel=False, k_shot_prior=1)
    [train_im, train_vox], target_vox = next(generator)
    #refiner = load_model('models/3_iter.h5')

    refined_vox = refiner.predict([train_im, train_vox])
    baseline_vox = baseline.predict([train_im, np.zeros_like(train_vox)])
    # refined_vox = refiner.predict([train_im, refined_vox])


    for i in range(num_points):
        print("Making Figure", i)
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        _ = ax.voxels((target_vox[i]>0.5).squeeze(), color='gray', edgecolor='k')
        ax.set_axis_off()
        ax.view_init(-65, 85)
        plt.savefig('examples/%s/ground_truth_%d.png' %(category,i))
        plt.close()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        _ = ax.voxels((refined_vox[i]>0.4).squeeze(), color="blue", edgecolor='k')
        ax.set_axis_off()
        ax.view_init(-65, 85)
        plt.savefig('examples/%s/pred_%d.png' %(category,i))
        plt.close()

        train_im[i] = shift_black_background_to_white(train_im[i])
        cv2.imwrite('examples/%s/image_%d.png' %(category,i), 255*train_im[i])
        plt.close()
        """
        np.save("examples/%s/new_%d.npy" %(category,i), (refined_vox[i]>0.4).squeeze())
        np.save("examples/%s/old_%d.npy" %(category,i), (baseline_vox[i]>0.4).squeeze())
        np.save("examples/%s/truth_%d.npy" %(category,i), (target_vox[i]).squeeze())
