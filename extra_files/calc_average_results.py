# coding: utf-8
import pickle; gen_results = pickle.load(open('finetuning_results_dict_generator.pickle', 'rb'))
whole_results = pickle.load(open('finetuning_results_dict_whole_model.pickle', 'rb'))
gen_results
categories = []
categories = ['benches', 'cabinets', 'lamps', 'rifles', 'sofas', 'vessels']
num_shots = [1, 2, 3, 4, 5, 10, 25]
import numpy as np
ave_gen_results = {n:np.average([gen_results[(category, n)] for category in categories]) for n in num_shots}
ave_whole_results = {n:np.average([whole_results[(category, n)] for category in categories]) for n in num_shots}
print(ave_gen_results)
print(ave_whole_results)
