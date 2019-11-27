import os
from sklearn.model_selection import train_test_split
from binvox_rw import read_as_3d_array
import numpy as np
import random
import pickle
import cv2
import re
from copy import copy
import h5py
from keras.preprocessing.image import apply_affine_transform
from keras.utils import to_categorical

base_im_dir = '/data/bw462/3d_recon/ShapeNetRendering/'
base_vox_dir = '/data/bw462/3d_recon/ShapeNetVox32/'
categories = [directory for directory in os.listdir(base_im_dir)  if '.' not in directory]

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

training_cat_ids = ["02691156", "02958343", "03001627", "03211117", 
                    "04401088", "03691459", "04379243"]
transfer_cat_ids = ["02828884", "02933112", "03636649", "04090263",
                    "04256520", "04530566"]


cat_id_to_index = {"02691156":0,
                     "02828884":1,
                     "02933112":2,
                     "02958343":3,
                     "03001627":4,
                     "03211117":5,
                     "03636649":6,
                     "04401088":7,
                     "04090263":8,
                     "04256520":9,
                     "03691459":10,
                     "04379243":11,
                     "04530566":12}


# including 4 because busses are subsetted into cars
shapenet_cats = [1, 3, 4, 5, 6, 8, 10]
novel_cats = [2, 7, 9]

def pascal_sampler(categories, n_per_yield=1, mode="train", prior=True, balance_classes=False):
    if categories == 'shapenet':
        cat_ids = shapenet_cats
    elif categories == 'novel':
        cat_ids = novel_cats
    elif categories == 'all':
        cat_ids = list(range(1, 11))
    elif type(categories)==int:
        cat_ids = [categories]
    else:
        cat_ids = categories
    if balance_classes:
        assert(n_per_yield % len(cat_ids) == 0)
        id_choices = []
        while len(id_choices) < n_per_yield:
            for cat_id in cat_ids:
                id_choices.append(cat_id)
        
    data_file = h5py.File('/data/bw462/pascal/dataset/PASCAL3D.mat', 'r')
    # Format is 3 x 100 x 100 x n
    # Need to check 100 dimension order
    all_images = data_file['image_{}'.format(mode)].value.transpose(3, 1, 2, 0)
    # Format is 27001 x n. First entry is id (1...10) then others are voxels that need to be reshaped
    voxel_dataset = data_file['model_{}'.format(mode)].value
    ids = voxel_dataset[0]
    # Need to think about sizing and making mine right for this
    all_voxels = voxel_dataset[1:].transpose(1, 0).reshape(-1, 30, 30, 30, 1)
    padded_voxels = np.zeros((all_voxels.shape[0], 32, 32, 32, 1))
    padded_voxels[:, 1:31, 1:31, 1:31, :] = all_voxels
    all_voxels = padded_voxels

    indices = np.in1d(ids, cat_ids)
    ids = ids[indices]
    all_voxels = all_voxels[indices]
    all_images = all_images[indices]

    id_to_im_vox_tuples = {i:[] for i in cat_ids}
    for model_id, im, vox in zip(ids, all_images, all_voxels):
        id_to_im_vox_tuples[model_id].append((cv2.resize(im, (127, 127)), vox))
    while True:
        if not balance_classes:
            id_choices = random.choices(cat_ids, k=n_per_yield)
        input_im_batch = []
        input_vox_batch = []
        target_vox_batch = []
        for cat_id in id_choices:
            input_im, target_vox = random.choice(id_to_im_vox_tuples[cat_id])
            if prior:
                _, input_vox = random.choice(id_to_im_vox_tuples[cat_id])
            else:
                input_vox = np.zeros((32, 32, 32, 1))
            if mode=='train':
                input_im = apply_affine_transform(input_im, theta=np.random.uniform(-30,30),
                                                  tx=np.random.uniform(-20, 20),
                                                  ty=np.random.uniform(-20, 20),
                                                  shear=np.random.uniform(-30, 30),
                                                  zx=np.random.uniform(0.8, 1.5),
                                                  zy=np.random.uniform(0.8, 1.5),
                                                  )
            input_im_batch.append(input_im)
            input_vox_batch.append(input_vox)
            target_vox_batch.append(target_vox)
        yield [np.array(input_im_batch), np.array(input_vox_batch)], np.array(target_vox_batch)
        
    


def old_split_ids(passphrase):
    # Don't want to accidentally resplit data
    assert passphrase == 'actually do this'
    mode_to_cat_to_model_ids = {mode:{} for mode in ["train", "val", "test"]}
    for category in categories:
        model_ids = os.listdir(base_im_dir + category)
        train_ids, non_train_ids = train_test_split(model_ids, test_size=0.1, random_state=1)
        val_ids, test_ids = train_test_split(non_train_ids, test_size=0.5, random_state=1)
        
        mode_to_cat_to_model_ids["train"][category] = train_ids
        mode_to_cat_to_model_ids["val"][category] = val_ids
        mode_to_cat_to_model_ids["test"][category] = test_ids

    pickle.dump(mode_to_cat_to_model_ids, open(base_im_dir + "../mode_to_cat_to_model_ids.pickle", "wb"))
    calc_and_save_average_category_voxels()

def r2n2_split_ids(passphrase):
    # Don't want to accidentally resplit data
    assert passphrase == 'actually do this'
    mode_to_cat_to_model_ids = {mode:{} for mode in ["train", "val", "test"]}
    for category in categories:
        model_ids = sorted(os.listdir(base_im_dir + category))
        num_models = len(model_ids)
        # R2N2 did 80-20 with no validation. Going to try this approach
        train_ids = model_ids[:int(0.75 * num_models)]
        val_ids = model_ids[int(0.75 * num_models):int(0.8 * num_models)]
        test_ids = model_ids[int(0.8 * num_models):]
        
        mode_to_cat_to_model_ids["train"][category] = train_ids
        mode_to_cat_to_model_ids["val"][category] = val_ids
        mode_to_cat_to_model_ids["test"][category] = test_ids

    pickle.dump(mode_to_cat_to_model_ids, open(base_im_dir + "../mode_to_cat_to_model_ids.pickle", "wb"))
    calc_and_save_average_category_voxels()
        
def calc_and_save_average_category_voxels():
    # Will do this from train
    cat_to_train_model_ids = pickle.load(open(base_im_dir + "../mode_to_cat_to_model_ids.pickle", "rb"))["train"]
    category_to_average_train_voxel = {}
    for category in categories:
        print("Calculating Average Vox for", category)
        summed_voxel = np.zeros((32, 32, 32))
        for voxel_id in cat_to_train_model_ids[category]:
            voxel_path = base_vox_dir + category + '/' + voxel_id + '/model.binvox'
            new_voxel = read_as_3d_array(open(voxel_path, 'rb')).data
            summed_voxel += new_voxel
        average_voxel = summed_voxel / len(cat_to_train_model_ids[category])
        category_to_average_train_voxel[category]  = average_voxel[..., np.newaxis]
    pickle.dump(category_to_average_train_voxel, open(base_im_dir + '../category_to_average_train_voxel.pickle', 'wb'))

    
def get_voxel_and_image_for_tuple(cat_and_id_tuple):
    category, model_id = cat_and_id_tuple
    voxel_path = base_vox_dir + category + '/' + model_id + '/model.binvox'
    im_path = base_im_dir + category + '/' + model_id + '/rendering/%02d.png' %np.random.randint(24)
    voxel = read_as_3d_array(open(voxel_path, 'rb')).data[..., np.newaxis]
    im = cv2.resize(cv2.imread(im_path), (127, 127)) / 255
    return (voxel, im )

def get_random_train_voxels_for_categories(model_category_list, cat_to_train_ids):
    ids = [random.choice(cat_to_train_ids[category]) for category in model_category_list]
    voxels = [read_as_3d_array(open(base_vox_dir + category + '/' + model_id + '/model.binvox', 'rb')).data[..., np.newaxis]
              for category,model_id in zip(model_category_list, ids)]
    return np.array(voxels)
    
def get_random_averaged_train_voxels_for_categories(model_category_list, cat_to_train_ids, num_shots=1):
    grouped_model_ids = [random.sample(cat_to_train_ids[category], k=num_shots) for category in model_category_list]
    voxels = []
    for category,id_list in zip(model_category_list, grouped_model_ids):
        averaged_shot = np.average([read_as_3d_array(open(base_vox_dir + category + '/' + model_id + '/model.binvox', 'rb')).data[..., np.newaxis]
                                  for model_id in id_list], axis=0)
        voxels.append(averaged_shot)
    return np.array(voxels)

def get_im(category, model_id):
    im_path = base_im_dir + category + '/' + model_id + '/rendering/%02d.png' %np.random.randint(24)
    im = cv2.resize(cv2.imread(im_path), (127, 127)) / 25
    return im
    
def image_category_generator(n_per_yield=1, mode="train"):
    mode_to_cat_to_model_ids = pickle.load(open(base_im_dir + "../mode_to_cat_to_model_ids.pickle", "rb"))
    cat_to_model_ids = mode_to_cat_to_model_ids[mode]
    cat_and_model_tuple_list = []
    for category, id_list in cat_to_model_ids.items():
        for model_id in id_list:
            cat_and_model_tuple_list.append((category, model_id))
    while True:
        cat_id_tuple_choices = random.choices(cat_and_model_tuple_list, k=n_per_yield)
        ims = [get_im(*info_tuple) for info_tuple in cat_id_tuple_choices]
        input_images = np.array(ims)
        target_indices = np.array( [cat_id_to_index[cat_id] for cat_id, _ in cat_id_tuple_choices] )
        targets = to_categorical(target_indices, 13)
        yield input_images, targets



def triple_generator(category_list=categories, n_per_yield=1, mode="train", input_average_voxel=True, k_shot_prior=0,
                     randomize_categories=False):
    assert (not (input_average_voxel and k_shot_prior))
    # If there are any letter than we need to cast to IDs
    if re.search('[a-zA-Z]', category_list[0]):
        category_list = [category_name_to_id[cat_name] for cat_name in category_list]
    
    mode_to_cat_to_model_ids = pickle.load(open(base_im_dir + "../mode_to_cat_to_model_ids.pickle", "rb"))
    cat_to_model_ids = mode_to_cat_to_model_ids[mode]
    if k_shot_prior:
        cat_to_train_ids = mode_to_cat_to_model_ids[mode]
    cat_to_average_train_voxel = pickle.load(open(base_im_dir + '../category_to_average_train_voxel.pickle', 'rb'))
    cat_and_model_tuple_list = []
    for category in category_list:
        id_list = cat_to_model_ids[category]
        for model_id in id_list:
            cat_and_model_tuple_list.append((category, model_id))
    while True:
        cat_id_tuple_choices = random.choices(cat_and_model_tuple_list, k=n_per_yield)
        vox_and_im_tuples = [get_voxel_and_image_for_tuple(info_tuple) for info_tuple in cat_id_tuple_choices]
        target_voxels, input_images = zip(*vox_and_im_tuples)
        target_voxels = np.array(target_voxels)
        input_images = np.array(input_images)
        if input_average_voxel:
            if randomize_categories:
                print("Shuffling") 
                model_category_list = random.choices(training_cat_ids, k=n_per_yield)
                input_voxels = np.array([cat_to_average_train_voxel[cat] for cat in model_category_list])
            else:
                input_voxels = np.array([cat_to_average_train_voxel[cat] for cat,_ in cat_id_tuple_choices])
        elif k_shot_prior:
            if randomize_categories:
                print("Shuffling")
                model_category_list = random.choices(transfer_cat_ids, k=n_per_yield)
            else:
                model_category_list, _ = zip(*cat_id_tuple_choices)
            input_voxels = get_random_averaged_train_voxels_for_categories(model_category_list, cat_to_train_ids, num_shots=k_shot_prior)
        else:
            input_voxels = np.zeros(target_voxels.shape)
        yield [input_images, input_voxels], target_voxels


def full_test_generator(category_list=categories, n_per_yield=1, mode="train", input_average_voxel=True, k_shot_prior=0,
                     randomize_categories=False):
    assert (not (input_average_voxel and k_shot_prior))
    mode = "test"
    # If there are any letter than we need to cast to IDs
    if re.search('[a-zA-Z]', category_list[0]):
        category_list = [category_name_to_id[cat_name] for cat_name in category_list]

    mode_to_cat_to_model_ids = pickle.load(open(base_im_dir + "../mode_to_cat_to_model_ids.pickle", "rb"))
    cat_to_model_ids = mode_to_cat_to_model_ids[mode]
    if k_shot_prior:
        cat_to_train_ids = mode_to_cat_to_model_ids[mode]
    cat_to_average_train_voxel = pickle.load(open(base_im_dir + '../category_to_average_train_voxel.pickle', 'rb'))
    cat_and_model_tuple_list = []
    for category in category_list:
        id_list = cat_to_model_ids[category]
        for model_id in id_list:
            cat_and_model_tuple_list.append((category, model_id))
    num_tuples = len(cat_and_model_tuple_list)
    for tuple_i, cat_and_model_tuple in enumerate(cat_and_model_tuple_list):
        print(tuple_i, '/', num_tuples)
        vox, input_images  = get_voxel_and_image_list_for_tuple(cat_and_model_tuple)
        target_voxels = np.array([vox for _ in input_images])
        input_images = np.array(input_images)
        if input_average_voxel:
            if randomize_categories:
                model_category_list = random.choices(training_cat_ids, k=24)
                input_voxels = np.array([cat_to_average_train_voxel[cat] for cat in model_category_list])
            else:
                cat = cat_and_model_tuple[0]
                input_voxels = np.array([copy(cat_to_average_train_voxel[cat]) for _ in range(24)])
        elif k_shot_prior:
            if randomize_categories:
                model_category_list = random.choices(transfer_cat_ids, k=24)
            else:
                model_category_list = [cat_and_model_tuple[0] for _ in input_images]
            input_voxels = get_random_averaged_train_voxels_for_categories(model_category_list, cat_to_train_ids, num_shots=k_shot_prior)
        else:
            input_voxels = np.zeros(target_voxels.shape)
        yield [input_images, input_voxels], target_voxels


def get_voxel_and_image_list_for_tuple(cat_and_id_tuple):
    category, model_id = cat_and_id_tuple
    voxel_path = base_vox_dir + category + '/' + model_id + '/model.binvox'
    im_paths = [base_im_dir + category + '/' + model_id + '/rendering/%02d.png' %im_index
                for im_index in range(24)]
    random.shuffle(im_paths)
    voxel = read_as_3d_array(open(voxel_path, 'rb')).data[..., np.newaxis]
    ims = [cv2.resize(cv2.imread(im_path), (127, 127)) / 255 for im_path in im_paths]
    return (voxel, ims)

               
def multiview_triple_generator(category_list=categories, n_per_yield=1, mode="train", input_average_voxel=True, k_shot_prior=0,
                     randomize_categories=False):
    assert (not (input_average_voxel and k_shot_prior))
    # If there are any letter than we need to cast to IDs
    if re.search('[a-zA-Z]', category_list[0]):
        category_list = [category_name_to_id[cat_name] for cat_name in category_list]

    mode_to_cat_to_model_ids = pickle.load(open(base_im_dir + "../mode_to_cat_to_model_ids.pickle", "rb"))
    cat_to_model_ids = mode_to_cat_to_model_ids[mode]
    if k_shot_prior:
        cat_to_train_ids = mode_to_cat_to_model_ids[mode]
    cat_to_average_train_voxel = pickle.load(open(base_im_dir + '../category_to_average_train_voxel.pickle', 'rb'))
    cat_and_model_tuple_list = []
    for category in category_list:
        id_list = cat_to_model_ids[category]
        for model_id in id_list:
            cat_and_model_tuple_list.append((category, model_id))
    while True:
        cat_id_tuple_choices = random.choices(cat_and_model_tuple_list, k=n_per_yield)
        vox_and_im_tuples = [get_voxel_and_image_list_for_tuple(info_tuple) for info_tuple in cat_id_tuple_choices]
        target_voxels, input_images = zip(*vox_and_im_tuples)
        target_voxels = np.array(target_voxels)
        input_images = np.array(input_images)
        if input_average_voxel:
            if randomize_categories:
                print("Shuffling")
                model_category_list = random.choices(training_cat_ids, k=n_per_yield)
                input_voxels = np.array([cat_to_average_train_voxel[cat] for cat in model_category_list])
            else:
                input_voxels = np.array([cat_to_average_train_voxel[cat] for cat,_ in cat_id_tuple_choices])
        elif k_shot_prior:
            if randomize_categories:
                print("Shuffling")
                model_category_list = random.choices(transfer_cat_ids, k=n_per_yield)
            else:
                model_category_list, _ = zip(*cat_id_tuple_choices)
            input_voxels = get_random_averaged_train_voxels_for_categories(model_category_list, cat_to_train_ids, num_shots=k_shot_prior)
        else:
            input_voxels = np.zeros(target_voxels.shape)
        yield [input_images, input_voxels], target_voxels

