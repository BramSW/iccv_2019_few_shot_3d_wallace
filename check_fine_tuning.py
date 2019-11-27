import pickle
import numpy as np

num_shots_arr = [1,2,3,4,5,10,25]
allowed_cats = ["benches", "cabinets", "lamps", "sofas", "rifles", "vessels"]
a = pickle.load(open("finetuning_results_dict_generator_complete.pickle", "rb"))


for num_shots in num_shots_arr:
    score = np.average([np.average(scores) for tup,scores in a.items() if tup[1]==num_shots and tup[0]!='phones'])
    print(num_shots, score)
