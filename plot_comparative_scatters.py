import matplotlib.pyplot as plt
import pickle
import numpy as np
from PIL import Image

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14.0


def crop_image(image):
    return image[70:240, 60:320]

#baseline_ious = pickle.load(open('models/gy_cats_RGB_baseline_histogram.pickle', 'rb'))
#new_2_iter_ious = pickle.load(open('models/gy_cats_2_iter_NONdelta_histogram.pickle', 'rb'))
#new_1_iter_ious = pickle.load(open('models/gy_cats_1_iter_1_shot_NONdelta_histogram.pickle', 'rb'))

model_names = ["Baseline", "3-Iter", "1-iter 1-shot"]

baseline_ious = pickle.load(open('test_performances/transfer_cats_testing_peformance_baseline.pickle', 'rb'))
three_iter_ious = pickle.load(open('test_performances/transfer_cats_testing_peformance_3_iter.pickle', 'rb'))
one_iter_one_shot_ious = pickle.load(open('test_performances/transfer_cats_testing_peformance_1_iter_1_shot.pickle', 'rb'))

plt.figure(0)
scatter_ax1 = plt.subplot2grid((2,3), (0,0))
scatter_ax2 = plt.subplot2grid((2,3), (0,1))
scatter_ax3 = plt.subplot2grid((2,3), (0,2))
im_ax_1 = plt.subplot2grid((2,3), (1,0))
im_ax_2 = plt.subplot2grid((2,3), (1,1))
im_ax_3 = plt.subplot2grid((2,3), (1,2))


category = "vessels"
cat_baseline_ious = baseline_ious[category][0]
cat_three_iter_ious = three_iter_ious[category][0]
cat_one_iter_one_shot_ious = one_iter_one_shot_ious[category][0]
iou_arr = [cat_baseline_ious, cat_three_iter_ious, cat_one_iter_one_shot_ious]

#image_texts = ["Baseline Pred.", "3-Iter Pred.", "Ground Truth"]

max_diff_index = np.subtract(cat_three_iter_ious, cat_baseline_ious).argmax()
print(cat_three_iter_ious[max_diff_index], cat_baseline_ious[max_diff_index])
#for col_i, category in enumerate(["cabinets", "vessels", "rifles"]):
#images_to_show = [crop_image(np.asarray(Image.open('max_diff_render_%s_%s.png' %(category, name))))
#                  for name in ['baseline', 'two_iter', 'truth']]

x_y_pairs = [(0,1), (0,2), (1,2)]
scatter_axes = [scatter_ax1, scatter_ax2, scatter_ax3]
for (x_i, y_i), ax in zip(x_y_pairs, scatter_axes):
    # ax.set_title(category)
    ax.scatter(iou_arr[x_i],iou_arr[y_i], s=1, marker='.', linewidths=0)
    ax.scatter([iou_arr[x_i][max_diff_index]], [iou_arr[y_i][max_diff_index]], s=15, color='red')
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='dashed', color='gray')
    #ax.text(0.5, -0.5, model_names[col_i], transform=ax.transAxes, horizontalalignment="center")
    #ax.text(1.25, 0.5, model_names[y_i], rotation=90, transform=ax.transAxes, verticalalignment="center")
    ax.set_xlabel(model_names[x_i] + " IoUs")
    ax.set_ylabel(model_names[y_i]+ " IoUs")

image_paths = ["scatter_imgs/vessels_%s.png" %s for s in ["baseline", "three_iter", "truth"]]
titles = ["Baseline Pred.", "3-Iteration Pred.", "True Shape"]
for im_path, ax, title in zip(image_paths, [im_ax_1, im_ax_2, im_ax_3], titles):
    ax.set_axis_off()
    im = crop_image(plt.imread(im_path))
    ax.imshow(im)
    ax.text(0.5, 1.0, title, transform=ax.transAxes, horizontalalignment="center")

plt.tight_layout()
plt.savefig("vessel_scatter_with_images.png")
