import pickle
import numpy as np


results = pickle.load(open('comparative_results_baseline_2iter_1iter1shot.pickle', 'rb'))
print("Baseline   2-iter    1-iter 1-shot")
for category, iou_arr in results.items():
    category = category[:1].upper() + category[1:]
    print(category, end=" & ")
    baseline = round(np.average(iou_arr[0][0]),3)
    iter_2 = round(np.average(iou_arr[1][1]),3)
    iter_1 = round(np.average(iou_arr[2][0]),3)
    print(baseline, end=" & ")
    print(iter_2, "(%0.3f)" %(iter_2 - baseline), end=" & ")
    print(iter_1, "(%0.3f)" %(iter_1 - baseline), "\\\\")
