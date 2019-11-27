import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
import sampler
import cv2
import sys
import numpy as np
from iou import iou

num_test_points = 1000
generator = sampler.triple_generator(n_per_yield=num_test_points, mode="val")
[train_im, train_vox], target_vox = next(generator)
refiner = load_model(sys.argv[1])
direct = load_model(sys.argv[2])
refined_vox = refiner.predict([train_im, train_vox])
from_image_vox = direct.predict([train_im, np.zeros(train_vox.shape)])


refined_ious = [iou(refined_vox[i], target_vox[i]) for i in range(num_test_points)]
direct_ious = [iou(from_image_vox[i], target_vox[i]) for i in range(num_test_points)]


plt.figure()
plt.plot(direct_ious, refined_ious, 'o')
plt.xlabel("Direct IoU")
plt.ylabel("Refiner IoU")
plt.plot([0,1],[0,1], linestyle='dashed')
plt.show()

plt.figure()
plt.hist(refined_ious, bins=np.linspace(0, 1.01, 0.1), alpha=0.5, label="Refined")
plt.hist(direct_ious, bins=np.linspace(0, 1.01, 0.1), alpha=0.5, label="Direct")
plt.legend()
plt.show()
plt.close()
