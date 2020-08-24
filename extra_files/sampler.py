from binvox_rw import read_as_3d_array
import numpy as np
import os
import pickle
from copy import copy
import random
import cv2
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Sequential
from keras import layers

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



def get_all_image_paths(category_list=category_name_to_id.keys(), mode="train", projection="orthographic"):
    all_image_paths = []
    for category in category_list:
        category_dir = "/data/bw462/3d_recon/data/%s_32x32alpha_%s_d100_r32_vp24_random_default/%s/" %(category, projection, mode)
        cat_image_keys = [f for f in os.listdir(category_dir) if '.json' not in f]
        cat_image_paths = [category_dir + key for key in cat_image_keys]
        all_image_paths.extend(cat_image_paths)
        print(category, len(cat_image_paths))
    return all_image_paths

def get_grouped_image_paths_with_voxels(mode="train"):
    image_paths = get_all_image_paths(mode=mode)
    category_to_model_to_data = {}
    for image_path in image_paths:
        category = image_path.split('/')[5].split('_')[0]
        model_id = image_path.split('/')[-1].split('_')[1]
        if category not in category_to_model_to_data:
            category_to_model_to_data[category] = {}
        if model_id not in category_to_model_to_data[category]:
            category_to_model_to_data[category][model_id] = [[],None]
            category_to_model_to_data[category][model_id][1] = match_single_image_path_to_voxel(image_path)
        category_to_model_to_data[category][model_id][0].append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255)
    return category_to_model_to_data

def get_n_multiimage_samples(n=1000, mode="val", empty_prior=True):
    category_to_model_to_data = get_grouped_image_paths_with_voxels(mode=mode)
    input_image_arrs = {cat:[] for cat in category_to_model_to_data}
    input_voxels = {cat:[] for cat in category_to_model_to_data}
    target_voxels = {cat:[] for cat in category_to_model_to_data}
    train_image_paths = get_all_image_paths(mode='train')
    category_to_voxel_paths = voxel_paths_hash_from_image_keys(train_image_paths)
    category_to_average_voxel = get_category_to_average_voxel(category_to_voxel_paths)
    for category in category_to_model_to_data:
        if empty_prior:
            input_voxel = np.zeros((32,32,32,1))
        else:
            input_voxel = category_to_average_voxel[category][...,np.newaxis]
        model_ids = random.sample(category_to_model_to_data[category].keys(), n)
        for model_id in model_ids:
            data_tuple = category_to_model_to_data[category][model_id]
            input_image_arrs[category].append(data_tuple[0])
            target_voxels[category].append(data_tuple[1])
            input_voxels[category].append(copy(input_voxel))
    return (input_image_arrs, input_voxels, target_voxels)



def all_voxel_paths_hash(category_list=category_name_to_id.keys()):
    category_to_voxel_paths = {}
    for category in category_list:
        category_dir = '/data/bw462/3d_recon/ShapeNetVox32/%s/' %category_name_to_id[category]
        cat_voxel_keys = os.listdir(category_dir)
        cat_voxel_paths = [category_dir + key + '/model.binvox' for key in cat_voxel_keys]
        category_to_voxel_paths[category] = cat_voxel_paths
    return category_to_voxel_paths


def voxel_paths_hash_from_image_keys(image_keys):
    category_to_voxel_paths = {category:[] for category in category_name_to_id}
    for image_key in image_keys:
        category = image_key.split('/')[5].split('_')[0]
        model_id = image_key.split('/')[-1].split('_')[1]
        category_to_voxel_paths[category].append("/data/bw462/3d_recon/ShapeNetVox32/%s/%s/model.binvox" %(category_name_to_id[category], model_id))
    for key,value in category_to_voxel_paths.items():
        category_to_voxel_paths[key] = list(set(value))
    return category_to_voxel_paths


def get_matching_category_voxel(image_path, category_to_voxel_paths):
    category = image_path.split('/')[5].split('_')[0]
    voxel_path = random.choice(category_to_voxel_paths[category])
    voxel = read_as_3d_array(open(voxel_path, 'rb')).data
    return voxel


def calc_average_voxel_from_list(voxel_path_list, spit_bad=False):
    # Going to use a running average soas not to wreck memory
    if not voxel_path_list: return None
    bad_paths = []
    ave_voxel = read_as_3d_array(open(voxel_path_list[0], 'rb')).data
    for i in range(1, len(voxel_path_list)):
        try:
            new_voxel = read_as_3d_array(open(voxel_path_list[i], 'rb')).data
        except ValueError:
            new_voxel = ave_voxel
            bad_paths.append(voxel_path_list[i])
            print("BAD", len(bad_paths), i)
        ave_voxel = (ave_voxel*i + new_voxel) / (i+1)
    if spit_bad: return ave_voxel, bad_paths
    else: return ave_voxel

def get_category_to_average_voxel(category_to_voxel_paths):
    return {category:calc_average_voxel_from_list(cat_list)
            for category,cat_list in category_to_voxel_paths.items()}

def get_overall_average_voxel(category_to_average_voxel):
    average_voxels = list(category_to_average_voxel.values())
    overall_average = sum(average_voxels) / len(average_voxels)
    return overall_average

def get_average_voxels(path_list, category_to_average_voxel):
    return np.array([category_to_average_voxel[path.split('/')[5].split('_')[0]]
                     for path in path_list])[...,np.newaxis]


def calc_average_shapenetsem_voxel_from_list(voxel_path_list, downsampling_model, spit_bad=False):
    # Going to use a running average soas not to wreck memory
    if not voxel_path_list: return None
    bad_paths = []
    ave_voxel = downsampling_model.predict(read_as_3d_array(open(voxel_path_list[0], 'rb')).data[np.newaxis, ..., np.newaxis])
    ave_voxel = ave_voxel.squeeze(axis=0)
    for i in range(1, len(voxel_path_list)):
        try:
            new_voxel = downsampling_model.predict(read_as_3d_array(open(voxel_path_list[i], 'rb')).data[np.newaxis, ..., np.newaxis])
        except ValueError:
            new_voxel = ave_voxel
            bad_paths.append(voxel_path_list[i])
            print("BAD", len(bad_paths), i)
        ave_voxel = (ave_voxel*i + new_voxel) / (i+1)
    ave_voxel = ave_voxel.squeeze(axis=0)
    if spit_bad: return ave_voxel, bad_paths
    else: return ave_voxel

def numpy_avg_round_pool3d(voxels, pool_size):
    old_voxel_dim = voxels.shape[1]
    new_voxel_dim = int(old_voxel_dim / pool_size)
    output = np.zeros((len(voxels), new_voxel_dim, new_voxel_dim, new_voxel_dim, 1))
    for i in range(len(voxels)):
        for j in range(0, new_voxel_dim):
            for k in range(0, new_voxel_dim):
                for l in range(0, new_voxel_dim):
                    sub_matrix = voxels[i,
                                        pool_size*j:pool_size*(j+1),
                                        pool_size*k:pool_size*(k+1),
                                        pool_size*l:pool_size*(l+1)]
                    output[i][j][k][l] = np.round(np.average(sub_matrix))
    return output
                    
def get_and_save_32_res_voxel(path, downsampling_model):
    res_32_path = path.split('.')[0] + '_32.npy'
    if os.path.exists(res_32_path):
        return np.load(res_32_path)
    else:
        high_res_voxel = read_as_3d_array(open(path, 'rb')).data[np.newaxis, ..., np.newaxis]
        low_res_voxel = np.squeeze(downsampling_model.predict(high_res_voxel), axis=0)
        np.save(res_32_path, low_res_voxel)
        return low_res_voxel

def get_and_save_128_res_im(path):
    res_128_path = path.split('.')[0] + '_128.png'
    if os.path.exists(res_128_path):
        return cv2.imread(res_128_path)
    else:
        im_128 = cv2.resize(cv2.imread(path), (128,128))
        cv2.imwrite(res_128_path, im_128)
        return im_128
    

def shapenetsem_triple_generator(category_list, n_per_yield, mode, input_average_voxel=True, resize_im_dim=128,
                                voxel_dim=32):
    assert (128/voxel_dim) == int(128/voxel_dim)
    pool_size = int(128/voxel_dim)
    resize_im_shape = (resize_im_dim, resize_im_dim)
    category_to_split_ids = pickle.load(open('/data/bw462/3d_recon/ShapeNetSem/category_to_split_ids.pickle', 'rb'))
    cat_to_mode_ids = {}
    cat_to_train_vox_paths = {}
    cat_and_id_tuples = []
    for category in category_list:
         if input_average_voxel:
              train_ids = category_to_split_ids[category]['train']
              cat_to_train_vox_paths[category] = ['/data/bw462/3d_recon/ShapeNetSem/models-binvox-solid/%s.binvox' %model_id
                                                  for model_id in train_ids]
         for model_id in copy(category_to_split_ids[category][mode]):
             path = '/data/bw462/3d_recon/ShapeNetSem/models-binvox-solid/%s.binvox' %model_id
             try:
                 _ = read_as_3d_array(open(path, 'rb'))
                 cat_and_id_tuples.append((category, model_id))
             except ValueError:
                 pass
            
    downsampling_model = Sequential()
    pool_size = 4
    downsampling_model.add(layers.MaxPooling3D(pool_size=(4,4,4), input_shape=(128, 128, 128,1)))
    downsampling_model.compile(optimizer='SGD', loss='mse')    
    
    if input_average_voxel:
        category_to_average_voxel = {}
        for category in cat_to_train_vox_paths:
            cat_ave_vox, bad_list = calc_average_shapenetsem_voxel_from_list(cat_to_train_vox_paths[category],
                                                                             downsampling_model,
                                                                             spit_bad=True)
            for bad_path in bad_list: cat_to_train_vox_paths[category].remove(bad_path)
            category_to_average_voxel[category] = cat_ave_vox
            
    print("Num Tuples:", len(cat_and_id_tuples))
    while True:
        # print("Making tuple choices")
        choices = random.choices(cat_and_id_tuples, k=n_per_yield)
        
        input_image_paths = ['/data/bw462/3d_recon/ShapeNetSem/screenshots/%s/%s-%d.png'
                        %(model_id, model_id, random.randint(0, 13))
                        for _, model_id in choices]
        # print("Loading images")
        input_images = np.array([get_and_save_128_res_im(path) for path in input_image_paths]) / 255
        # print("Choosing and loading voxels")
        target_voxel_paths = ['/data/bw462/3d_recon/ShapeNetSem/models-binvox-solid/%s.binvox' %model_id
                              for _, model_id in choices]
        target_voxels = np.array([get_and_save_32_res_voxel(path, downsampling_model) for path in target_voxel_paths])
        # print(target_voxels.shape)
        if input_average_voxel:
            input_voxels = np.array([category_to_average_voxel[category] for category, _ in choices])
        else:
            input_voxels = np.zeros(target_voxels.shape)
        yield ([input_images, input_voxels], target_voxels)

        

       


def shapenetsem_category_triple_generator(category_list, n_per_yield, mode, resize_dim=128, yield_categories=[]):
    category_to_one_hot = {}
    num_categories = len(category_list)
    for index, category in enumerate(category_list):
        one_hot_vec = np.zeros(num_categories)
        one_hot_vec[index] = 1
        category_to_one_hot[category] = one_hot_vec
        
    resize_shape = (resize_dim, resize_dim)
    category_to_split_ids = pickle.load(open('/data/bw462/3d_recon/ShapeNetSem/category_to_split_ids.pickle', 'rb'))
    cat_to_mode_ids = {}
    cat_to_train_vox_paths = {}
    cat_and_id_tuples = []
    if not yield_categories:
        yield_categories = category_list
    for category in yield_categories:
         for model_id in copy(category_to_split_ids[category][mode]):
             path = '/data/bw462/3d_recon/ShapeNetSem/models-binvox-solid/%s.binvox' %model_id
             try:
                 _ = read_as_3d_array(open(path, 'rb'))
                 cat_and_id_tuples.append((category, model_id))
             except ValueError:
                 pass
    while True:
        choices = random.choices(cat_and_id_tuples, k=n_per_yield)
        
        input_image_paths = ['/data/bw462/3d_recon/ShapeNetSem/screenshots/%s/%s-%d.png'
                        %(model_id, model_id, random.randint(0, 13))
                        for _, model_id in choices]
        
        input_images = np.array([cv2.resize(cv2.imread(path), resize_shape) for path in input_image_paths]) / 255
        target_voxel_paths = ['/data/bw462/3d_recon/ShapeNetSem/models-binvox-solid/%s.binvox' %model_id
                              for _, model_id in choices]
        target_voxels = np.array([read_as_3d_array(open(path, 'rb')).data for path in target_voxel_paths])[..., np.newaxis] 
        batch_input_codes = np.array([category_to_one_hot[category]
                                      for category,_ in choices])          
        yield ([input_images, batch_input_codes], target_voxels)

def triple_generator(category_list=category_name_to_id.keys(), n_per_yield=1, mode="train", input_average_voxel=True, category_agnostic_input_average=False):
    ##### Need to make train-val-test split for these
    ##### Also, images are 137x137 atm. Decide whether you care (probably nah)
    ##### Change the below for the new renderings
    train_image_paths = get_all_image_paths(category_list, mode="train")
    all_image_paths = get_all_image_paths(category_list, mode=mode)
    # category_to_voxel_paths = all_voxel_paths_hash(category_list=category_list)
    ##### Below function to account for new image paths
    category_to_voxel_paths = voxel_paths_hash_from_image_keys(all_image_paths)
    if input_average_voxel: 
        train_category_to_voxel_paths = voxel_paths_hash_from_image_keys(train_image_paths)
        category_to_average_voxel = get_category_to_average_voxel(train_category_to_voxel_paths)
        # overall_average_voxel = get_overall_average_voxel(category_to_average_voxel)
    while True:
        batch_im_paths = random.choices(all_image_paths, k=n_per_yield)
        batch_images = image_paths_to_images(batch_im_paths)[...,np.newaxis]
         
        if category_agnostic_input_average:
             batch_input_voxels = np.array( [overall_average_voxel] * n_per_yield)
        elif input_average_voxel:
              batch_input_voxels = get_average_voxels(batch_im_paths, category_to_average_voxel) 
        else:
             # Get matching category voxel will need to be updated once it's used
             batch_input_voxels = np.array([get_matching_category_voxel(im_path, category_to_voxel_paths)
                                   for im_path in batch_im_paths])[...,np.newaxis]
        batch_target_voxels = match_image_paths_to_voxels(batch_im_paths)[...,np.newaxis]
        yield ([batch_images, batch_input_voxels], batch_target_voxels)

            
      

            
def triple_category_generator(category_list=category_name_to_id.keys(), n_per_yield=1, mode="train", yield_categories=[]):
    category_to_one_hot = {}
    num_categories = len(category_list)
    for index, category in enumerate(category_list):
        one_hot_vec = np.zeros(num_categories)
        one_hot_vec[index] = 1
        category_to_one_hot[category] = one_hot_vec
    if yield_categories:
        all_image_paths = get_all_image_paths(yield_categories, mode=mode)
    else:
        all_image_paths = get_all_image_paths(category_list, mode=mode)
    category_to_voxel_paths = voxel_paths_hash_from_image_keys(all_image_paths)
    while True:
        batch_im_paths = random.choices(all_image_paths, k=n_per_yield)
        batch_images = image_paths_to_images(batch_im_paths)[...,np.newaxis]
         
        batch_input_categories = [im_path.split('/')[5].split('_')[0]
                                  for im_path in batch_im_paths]
        batch_input_codes = np.array([category_to_one_hot[category]
                                      for category in batch_input_categories])
        batch_target_voxels = match_image_paths_to_voxels(batch_im_paths)[...,np.newaxis]
        yield ([batch_images, batch_input_codes], batch_target_voxels)
            

def blanks_generator(category_list, n_per_yield=1, mode="train", input_average_voxel=True):
    # Blanks refers to the input voxels being blank
    all_image_paths = get_all_image_paths(category_list, mode=mode)
    # category_to_voxel_paths = all_voxel_paths_hash(category_list=category_list)
    category_to_voxel_paths = voxel_paths_hash_from_image_keys(all_image_paths)
    while True:
         batch_im_paths = random.choices(all_image_paths, k=n_per_yield)
         batch_images = image_paths_to_images(batch_im_paths)[...,np.newaxis]

         batch_target_voxels = match_image_paths_to_voxels(batch_im_paths)[...,np.newaxis]
         batch_input_voxels = np.zeros(batch_target_voxels.shape)
         yield ([batch_images, batch_input_voxels], batch_target_voxels)




########################################
def n_random_voxel_paths_in_category(n, category_id):
    category_dir = '/data/bw462/3d_recon/ShapeNetVox32/%s/' %category_id
    voxel_keys = os.listdir(category_dir)
    chosen_voxel_paths = [category_dir + random.choice(voxel_keys) + '/model.binvox' for _ in range(n)]
    return chosen_voxel_paths

def n_random_voxels_in_category(n, category_id):
    paths = n_random_voxel_paths_in_category(n, category_id)
    voxels = np.array([read_as_3d_array(open(path, 'rb')).data for path in paths])
    return voxels

def n_random_image_paths_in_category(n, category_name, subdir="train"):
    category_dir = "/data/bw462/3d_recon/data/%s_32x32alpha_orthographic_d100_r32_vp24_random_default/%s/" %(category_name, subdir)
    image_keys = [f for f in os.listdir(category_dir) if '.json' not in f]
    chosen_keys = random.sample(image_keys, min(n, len(image_keys)))
    chosen_image_paths = [category_dir + key for key in chosen_keys]
    return chosen_image_paths



def image_paths_to_images(image_paths):
    images = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths])/255
    return images



#def n_random_images_in_category(n, category_name, subdir="train"):
#    paths = n_random_image_paths_in_category(n, category_name, subdir=subdir)
#    images = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in paths])/255
#    return images




def match_single_image_path_to_voxel(image_path):
    image_path_parts = image_path.split('/')[-1].split('_')
    category_id = image_path_parts[0]
    shape_id = image_path_parts[1]
    voxel_path = "/data/bw462/3d_recon/ShapeNetVox32/%s/%s/model.binvox" %(category_id, shape_id)
    voxel = read_as_3d_array(open(voxel_path, 'rb')).data
    return voxel

def match_image_paths_to_voxels(paths):
    return np.array([match_single_image_path_to_voxel(path) for path in paths])


def construct_and_save_split_data(n_per_category):
    category_list = list(category_name_to_id.keys())
    base_dir = '/data/bw462/3d_recon/standard_split/'
    train_dir = base_dir + 'train/'
    val_dir = base_dir + 'val/'
    test_dir = base_dir + 'test/'

    for i, category in enumerate(category_list):
        print("Loading", i+1, "out of", len(category_list), "categories (%s)" %category)
        for mode in ["train", "val", "test"]:
            print("Creating", mode)
            cat_image_paths = n_random_image_paths_in_category(n_per_category, category, subdir=mode)
            cat_input_images = image_paths_to_images(cat_image_paths)
            cat_input_voxels = n_random_voxels_in_category(n_per_category, category_name_to_id[category])
            cat_target_voxels = match_image_paths_to_voxels(cat_image_paths)
         
            zipped_triples = list(zip(cat_input_images, cat_input_voxels, cat_target_voxels))
            random.shuffle(zipped_triples)
            cat_input_images, cat_input_voxels, cat_target_voxels = zip(*zipped_triples)
        
            cat_input_images = np.array(cat_input_images).reshape(-1, 32, 32, 1)
            cat_input_voxels = np.array(cat_input_voxels).reshape(-1, 32, 32, 32, 1)
            cat_target_voxels = np.array(cat_target_voxels).reshape(-1, 32, 32, 32, 1)
            pickle.dump((cat_input_images, cat_input_voxels, cat_target_voxels), open(base_dir + mode + "/" + category + "_triples.pickle", "wb"))
    return

def load_inputs_and_targets(categories=category_name_to_id.keys(), mode='train', num_points=None):
    base_dir = "/data/bw462/3d_recon/standard_split/%s/" %mode
    input_images = []
    input_voxels = []
    target_voxels = []
    for category in categories:
        cat_input_images, cat_input_voxels, cat_target_voxels = pickle.load(open(base_dir + category + "_triples.pickle", "rb"))
        input_images.extend(cat_input_images)
        input_voxels.extend(cat_input_voxels)
        target_voxels.extend(cat_target_voxels)

    if num_points is None: num_points = len(input_images)
    zipped_triples = list(zip(input_images, input_voxels, target_voxels))
    random.shuffle(zipped_triples)
    zipped_triples = zipped_triples[:num_points]
    input_images, input_voxels, target_voxels = zip(*zipped_triples)
    input_images = np.array(input_images)
    input_voxels = np.array(input_voxels)
    target_voxels = np.array(target_voxels)
    return input_images, input_voxels, target_voxels

if __name__=='__main__':
    construct_and_save_split_data(10000)
