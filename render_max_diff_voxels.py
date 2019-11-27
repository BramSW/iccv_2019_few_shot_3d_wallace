import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from binvox_rw import read_as_3d_array
from keras.models import load_model

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


base_im_dir = '/data/bw462/3d_recon/ShapeNetRendering/'
base_vox_dir = '/data/bw462/3d_recon/ShapeNetVox32/'

baseline_model = load_model('models/baseline.h5')
three_iter_model = load_model('models/3_iter.h5')
one_iter_one_shot_model = load_model('models/1_iter_1_shot.h5')

baseline_ious = pickle.load(open('test_performances/transfer_cats_testing_peformance_baseline.pickle', 'rb'))
three_iter_ious = pickle.load(open('test_performances/transfer_cats_testing_peformance_3_iter.pickle', 'rb'))
one_iter_one_shot_ious = pickle.load(open('test_performances/transfer_cats_testing_peformance_1_iter_1_shot.pickle', 'rb'))
category_to_prior_vox = pickle.load(open(base_im_dir + '../category_to_average_train_voxel.pickle', 'rb'))

categories = ["rifles", "vessels"]

cat_to_model_ids = pickle.load(open(base_im_dir + '../mode_to_cat_to_model_ids.pickle', 'rb'))['test']

for category in categories:
    cat_baseline_ious = baseline_ious[category][0]
    cat_three_iter_ious = three_iter_ious[category][0]
    cat_one_iter_one_shot_ious = one_iter_one_shot_ious[category][0]
    iou_arr = [cat_baseline_ious, cat_three_iter_ious, cat_one_iter_one_shot_ious]
    
    max_diff_index = np.subtract(cat_three_iter_ious, cat_baseline_ious).argmax()
    category_id = category_name_to_id[category]
    model_ids = cat_to_model_ids[category_id]
    max_diff_model_index = max_diff_index // 24
    max_diff_model_id = model_ids[max_diff_model_index]
    max_diff_image_index = max_diff_index % 24
    im_path = base_im_dir + category_id + '/' + max_diff_model_id + '/rendering/%02d.png' %max_diff_image_index
    input_im = (cv2.resize(cv2.imread(im_path), (127, 127)) / 255)[np.newaxis, ...]
    voxel_path = base_vox_dir + category_id + '/' + max_diff_model_id + '/model.binvox'
    target_vox = read_as_3d_array(open(voxel_path, 'rb')).data[..., np.newaxis]
    prior_vox = category_to_prior_vox[category_id][np.newaxis, ...]
    cv2.imwrite("max_diff_image_%s.png" %category, input_im.squeeze()*255)
    baseline_pred = baseline_model.predict([input_im, np.zeros((1,32,32,32,1))])
    three_iter_pred = three_iter_model.predict([input_im, prior_vox])
    one_iter_pred = one_iter_one_shot_model.predict([input_im, prior_vox])
    np.save("voxels/baseline_pred_%s.npy" %category, baseline_pred)
    np.save("voxels/three_iter_pred_%s.npy" %category, three_iter_pred)
    np.save("voxels/one_iter_pred_%s.npy" %category, one_iter_pred)
    preds = [baseline_pred, three_iter_pred, one_iter_pred]
    names = ["baseline", "three_iter", "one_iter_one_shot"]
    for pred,name in zip(preds,names):
        fig =plt.figure()
        ax = fig.gca(projection='3d')
        ax.view_init(-65, 85)
        _ = ax.voxels(pred.squeeze()>0.4, color='gray', edgecolor='k')
        ax.set_axis_off()
        plt.savefig('max_diff_render_%s_%s.png' %(category, name))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(-65, 85)
    _ = ax.voxels(target_vox.squeeze(), color='gray', edgecolor='k')
    ax.set_axis_off()
    plt.savefig('max_diff_render_%s_truth.png' %category)
    np.save("voxels/ground_truth_%s.npy" %category, target_vox)


