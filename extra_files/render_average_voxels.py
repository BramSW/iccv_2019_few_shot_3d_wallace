import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
import sampler
import cv2
import sys
import numpy as np
import pickle

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
category_id_to_name = {val:key for key,val in category_name_to_id.items()}


cat_to_ave_voxel = pickle.load(open('/data/bw462/3d_recon/category_to_average_train_voxel.pickle', 'rb'))

fig, axes = plt.subplots(4,4, subplot_kw={'projection':'3d'})

axes_indices = [(i,j) for i in range(4) for j in range(4)]

for (i,j), cat_name in zip(axes_indices, sorted(category_name_to_id.keys())):
    cat_id = category_name_to_id[cat_name]
    vox = cat_to_ave_voxel[cat_id]
    ax = axes[i,j]
    ax.voxels((vox>0.4).squeeze(), color='gray', edgecolor='k')
    ax.set_title(cat_name + "(t=0.4)", fontsize=10)
    ax.view_init(-65, 85)
    ax.set_axis_off()

repeated_cat_names = ["sofas", "speakers", "tables"]
for j, cat_name in zip(range(1,4), repeated_cat_names):
    cat_id = category_name_to_id[cat_name]
    vox = cat_to_ave_voxel[cat_id]
    ax = axes[3,j]
    ax.voxels((vox>0.2).squeeze(), color='gray', edgecolor='k')
    ax.set_title(cat_name + "(t=0.2)", fontsize=10)
    ax.view_init(-65, 85)
    ax.set_axis_off()
plt.savefig("average_voxels.png")
