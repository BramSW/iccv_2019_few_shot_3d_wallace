import keras
import numpy as np
import highres_sampler
from iou import iou
import matplotlib.pyplot as plt

model = keras.models.load_model('/data/bw462/3d_recon/models/1_iter_1_shot.h5')

for category in ["benches", "cabinets", "lamps", "vessels", "sofas", "rifles"]:
    print(category)
    generator = highres_sampler.triple_generator(category_list=[category], n_per_yield=100, mode="test", input_average_voxel=False,
                                                            k_shot_prior=1)

    [input_ims, input_voxels], target_voxels = next(generator)

    stds = []
    for i in range(100):
        ious = []
        for j in range(100):
            result = model.predict([input_ims[i:i+1], input_voxels[j:j+1]])
            ious.append(iou(result, target_voxels[i:i+1], threshold=0.4))
        # print(ious)
        # print(np.std(ious))
        stds.append(np.std(ious))

    print(category, np.average(stds))
