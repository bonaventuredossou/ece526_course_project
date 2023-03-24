from utils import *
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import shutil
from typing import List, Tuple

# data source https://challenge.isic-archive.com/data/#2016
# build datasets splits
def create_train_val_splits(path: str)-> None:
    # Splits the training set into train and validation
    dataset = pd.read_csv(path)
    label_map = {'benign': 0, 'malignant': 1}
    dataset['labels'] = dataset['labels'].apply(lambda x: label_map[x])
    train, val = train_test_split(dataset, stratify=dataset['labels'], train_size=0.8, random_state=1234)

    train.to_csv('../train_split.csv', index=False)
    val.to_csv('../val_split.csv', index=False)

# only for the first time to generate the splits
# train_path = '../ISBI2016_Training_Labels.csv'
# create_train_val_splits(train_path)

def make_dirs()-> None:
    directory = os.path.join('../data')
    for split in ['train', 'val', 'test']:
        if not os.path.exists(os.path.join(directory, split)):
            os.mkdir(os.path.join(directory, split))
        split_dir = os.path.join(directory, split)
        for category in range(2):
            if not os.path.exists(os.path.join(split_dir, str(category))):
                os.mkdir(os.path.join(split_dir, str(category)))

make_dirs()

def create_images_to_dir(dataset_split: str, dataset_path: str)-> None:
    data_paths, data_labels = get_image_and_labels(dataset_path)
    directory = os.path.join('../','data', dataset_split)
    if dataset_split in ['train', 'val']:
        base_path = '../ISBI2016_Training_Data'
    else:
        base_path = '../ISBI2016_Test_Data'
    total_data = len(data_paths)
    for index in tqdm(range(total_data), desc ="Data Creation Progress"):
        data_path, data_label = os.path.join(base_path, '{}.jpg'.format(data_paths[index])), data_labels[index]
        output_directory = os.path.join(directory, str(data_label))
        shutil.copy(data_path, os.path.join(output_directory, '{}.jpg'.format(data_paths[index])))


def get_image_and_labels(path: str) -> Tuple[List, List]:
    dataset = pd.read_csv(path)
    images = dataset.images.tolist()
    labels = dataset.labels.tolist()
    return images, labels

# Run only for the first time to create the folders and data
# training_path = '../train_split.csv'
# eval_path = '../val_split.csv'
# test_path = '../test_split.csv'

# create_images_to_dir('train', training_path)
# create_images_to_dir('val', eval_path)
# create_images_to_dir('test', test_path)

data_dir = '../data'
batch_size = 8

dataloader_dict = preprocessing(data_dir, batch_size)