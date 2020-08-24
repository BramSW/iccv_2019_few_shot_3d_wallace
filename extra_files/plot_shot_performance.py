import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16.0

num_shots = [1, 2, 3, 4, 5, 10, 25]
ious_1_iter = [0.384, 0.390, 0.393, 0.394, 0.395, 0.397, 0.399]
whole_model_finetune = [0.378, 0.379, 0.382, 0.387, 0.388, 0.393, 0.406]
generator_finetune = [0.372, 0.378, 0.382, 0.382, 0.385, 0.394, 0.400]
plt.figure()
plt.plot(num_shots, ious_1_iter, label="1-Iteration 1-Shot Model")
plt.plot(num_shots, generator_finetune, label="Finetune Generator")
plt.plot(num_shots, whole_model_finetune, label="Finetune Whole Model")
plt.hlines(0.361, xmin=1, xmax=25, label="Image-Only Baseline")
plt.legend(loc=(0.22, 0.1))
plt.xlabel("# Shots")
plt.ylabel("Category-Wise IoU")
plt.tight_layout()
plt.savefig("num_shots_vs_iou.png")


