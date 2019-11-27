import matplotlib.pyplot as plt
import pickle
import numpy as np

#baseline_ious = pickle.load(open('models/gy_cats_RGB_baseline_histogram.pickle', 'rb'))
#new_2_iter_ious = pickle.load(open('models/gy_cats_2_iter_NONdelta_histogram.pickle', 'rb'))
#new_1_iter_ious = pickle.load(open('models/gy_cats_1_iter_1_shot_NONdelta_histogram.pickle', 'rb'))

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14.0


baseline_ious = pickle.load(open('test_performances/transfer_cats_testing_peformance_baseline.pickle', 'rb'))
iter_3_ious = pickle.load(open('test_performances/transfer_cats_testing_peformance_10_shot_3_iter.pickle', 'rb'))
iter_1_ious = pickle.load(open('test_performances/transfer_cats_testing_peformance_10_shot_1_iter_1_shot.pickle', 'rb'))

indices_to_use = [ 0, 0, 0 ]
model_names = ["Baseline", "3-Iter", "1-iter 1-shot"]

# category_to_ious = pickle.load(open('comparative_results_baseline_2iter_1iter1shot.pickle' ,'rb'))

fix, axes = plt.subplots(1,3, sharey=True)
for col_i, category in enumerate(["cabinets", "rifles", "vessels"]):
    ax = axes[col_i]
    iou_arr = [ious[category] for ious in [baseline_ious, iter_3_ious, iter_1_ious]]
    sorted_ious = [sorted(model_ious[0][:]) for model_ious in iou_arr]
    for ious,name in zip(sorted_ious, model_names):
        ax.plot(ious, label=name, linewidth=2)
    ax.set_ylabel("IoU")
    ax.set_xlabel("Index")
    if col_i == 2: ax.legend()
    ax.set_title(category)
    ax.set_xticks(range(0, len(ious), 1000), minor=True)
    ax.set_yticks(np.arange(0, 1.01, 0.1), minor=True)
    """
    axes[0, col_i].text(0.5, 1.4, category, transform=axes[0, col_i].transAxes)
    for row_i in range(3):
        ax = axes[row_i, col_i]
        index_to_use = indices_to_use[row_i]
        ious = iou_arr[row_i][index_to_use]
        ax.hist(ious, bins=np.arange(0, 1.0001, 0.04), edgecolor='k', normed=True)
        ax.set_title("Average IoU: %0.3f" %np.average(ious))
        if col_i == 2:
            ax.text(1.1, 0.5, model_names[row_i], rotation=270, transform=ax.transAxes)
    ax.set_xlabel("IoU")
    """
plt.tight_layout()
plt.savefig("category_cdf.png")
