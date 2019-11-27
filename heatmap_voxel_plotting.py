import matplotlib.pyplot as plt
import numpy as np
from binvox_rw import read_as_3d_array
from mpl_toolkits.mplot3d import Axes3D
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


def explode(data):
    data_shape = data.shape
    if data_shape == (32,32,32):
        size = np.array(data.shape)*2
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
    elif data_shape == (32,32,32,4):
        data_e = np.zeros( (63,63,63,4), dtype=data.dtype)
        data_e[::2, ::2, ::2, :] = data
    return data_e

# build up the numpy logo
#n_voxels = np.zeros((4, 3, 4), dtype=bool)
#n_voxels = read_as_3d_array(open('model.binvox', 'rb')).data
done_cat_to_angles = {"airplanes":(-65, 85), 
                      "benches":(120, 240),
                      "cabinets":(180, 40),
                      "cars":(-65, 85),
                      "chairs": (300, 240),
                      "displays":(160, 320),
                      "lamps": (80, 280),
                      "phones": (120, 320),
                      "sofas": (160, 200),
                      "speakers": (340, 0),
                      "tables": (120, 100),
                      "rifles":(-65, 85),
                      "vessels":(40, 240)}

for cat,cat_id in category_name_to_id.items():
    if cat in done_cat_to_angles: elevation_azimuth_tuples = [done_cat_to_angles[cat]]
    else: elevation_azimuth_tuples = [(elev, azim) for elev in range(0, 360, 20) for azim in range(0, 360, 20)]
    n_voxels = pickle.load(open('/data/bw462/3d_recon/category_to_average_train_voxel.pickle', 'rb'))[cat_id].squeeze()

    #n_voxels[0, 0, :] = True
    #n_voxels[-1, 0, :] = True
    #n_voxels[1, 0, 2] = True
    #n_voxels[2, 0, 1] = True
    facecolors = np.empty((32,32,32,4))
    edgecolors = np.zeros((32,32,32,4))

    for i in range(32):
        for j in range(32):
            for k in range(32):
                if n_voxels[i][j][k] > 0.9:
                    facecolors[i][j][k] = [1,0,0,0.1] # red
                elif n_voxels[i][j][k] > 0.6:
                    facecolors[i][j][k] = [1,1,0,0.1] # yellow
                elif n_voxels[i][j][k] > 0.3:
                    facecolors[i][j][k] = [0,0,1,0.1] # blue
                else:
                    facecolors[i][j][k] = [1,1,1,0.0001]

    #facecolors = np.where(n_voxels>0.4, 'red', '#7A88CCC0')
    filled = (n_voxels>0.3) # np.ones(n_voxels.shape)
    """
    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.voxels(filled, facecolors=facecolors)
    plt.show()
    """
    #############

    # upscale the above voxel image, leaving gaps
    filled_2 = explode(filled)
    fcolors_2 = explode(facecolors)
    ecolors_2 = explode(edgecolors)

    # Shrink the gaps
    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95
    for elevation, azimuth in elevation_azimuth_tuples:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
            # ax.view_init(-65,85) # planes, cars, rifles
            ax.view_init(elevation, azimuth) # 
            #ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2, alpha=0.1)
            ax.set_axis_off()
            plt.savefig('heatmap_voxels/' + cat + '_heatmap_ave_%d_%d.png' %(elevation, azimuth))
            print(elevation, azimuth, cat)
            plt.close()
