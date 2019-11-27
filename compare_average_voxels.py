import pickle
import numpy as np
from iou import iou 


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
training_cats = ["airplanes", "cars", "chairs", "displays", "phones", "speakers", "tables"]
training_ids = [category_name_to_id[cat] for cat in training_cats]
transfer_cats = ["benches", "cabinets", "lamps", "rifles", "sofas", "vessels"]
transfer_ids = [category_name_to_id[cat] for cat in transfer_cats]

cat_to_ave_voxel = pickle.load(open('/data/bw462/3d_recon/category_to_average_train_voxel.pickle', 'rb'))

for cat_id_transfer in transfer_ids:
    differences = []
    ious = []
    for cat_id_train in training_ids:
        vox_1 = cat_to_ave_voxel[cat_id_transfer]
        vox_2 = cat_to_ave_voxel[cat_id_train]
        squared_diff_matrix = (vox_1 - vox_2) ** 2
        differences.append(np.sum(squared_diff_matrix))
        ious.append(iou(vox_1, vox_2))
    # print(category_id_to_name[cat_id_transfer], "Min:", np.min(differences), "Max:", np.max(differences), "Average:", np.average(differences))
    print(category_id_to_name[cat_id_transfer], sorted(ious))
