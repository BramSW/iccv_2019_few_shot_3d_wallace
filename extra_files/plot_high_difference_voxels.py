import pickle
import numpy as np
import matplotlib.pyplot as plt

category_to_ious = pickle.load(open('comparative_results_baseline_2iter_1iter1shot.pickle' ,'rb'))

baseline_i = 0
iter_2_i = 1

for category in ["rifles", "vessels"]:
    baseline_ious = np.subtract(category_to_ious[category][1][0], category_to_ious[category][0][0])
    print(category, baseline_ious.max(), baseline_ious.argmax())
