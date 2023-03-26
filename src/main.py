from utils import *
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import shutil
from typing import List, Tuple
import torch.optim as optim

num_gpus = [i for i in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data source https://challenge.isic-archive.com/data/#2016
# build datasets splits
def create_train_val_splits(path: str)-> None:
    # Splits the training set into train and validation
    dataset = pd.read_csv(path)
    label_map = {'benign': 0, 'malignant': 1}
    dataset['labels'] = dataset['labels'].apply(lambda x: label_map[x])
    train, val = train_test_split(dataset, stratify=dataset['labels'], train_size=0.66, random_state=1234)

    train.to_csv('../train_split.csv', index=False)
    val.to_csv('../val_split.csv', index=False)

# only for the first time to generate the splits ~600 images for training, and 300 for evaluation
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

# Run only for the first time
# make_dirs()

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

def build_model() -> BasicCNN:
    model = BasicCNN()
    if len(num_gpus) > 1:
        print("Let's use", len(num_gpus), "GPUs!")
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in num_gpus)
        model = torch.nn.DataParallel(model, device_ids=num_gpus)
        model = model.module
    model = model.to(device)
    return model

# used to perform MC-Dropout
def set_dropout(trained_model):
    trained_model.eval()
    for name, module in trained_model.named_modules():
        if 'dropout' in name:
            module.train()

def query_pool(model: BasicCNN, dataloaders: Dict, strategy: str,
               batch_size: int, query_size: int) -> Dict:
    # set the model's dropout unit in training mode
    set_dropout(model)
    # run the model on the pool
    predictions, labels = run_uncertainty(model, dataloaders['pool'])
    entropy, expectation_entropy = compute_entropy(predictions)
    if strategy == 'max_entropy':
        top_indices = compute_max_entropy(entropy)
    if strategy == 'mean_std':
        mean_std, top_indices = compute_mean_std(predictions)
    if strategy == 'bald':
        mutual_information, top_indices = compute_bald(entropy, expectation_entropy)
    
    top_indices = top_indices.tolist()
    
    print('Initial size: Train = {}, Pool = {}'.format(dataloaders['train'], dataloaders['pool']))
    training_dataset = dataloaders['train'].dataset
    pool_dataset = dataloaders['pool'].dataset

    # add indices to the training set
    for index in top_indices:
        training_dataset.imgs.append(pool_dataset.imgs[index])

    # removes indices from the pool set    
    for index in top_indices:
        del pool_dataset.imgs[index]

    # reaasigns the updated data loaders
    dataloaders['train'] = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    dataloaders['pool'] = DataLoader(pool_dataset, batch_size=batch_size, shuffle=False)

    print('Size after Update: Train = {}, Pool = {}'.format(dataloaders['train'],
                                                            dataloaders['pool']))

    return dataloaders

def run_strategy(strategy_name: str) -> None:

    data_dir = '../data'
    batch_size, num_epochs, lr = 8, 100, 1e-4
    dataloader_dict = preprocessing(data_dir, batch_size, strategy_name)
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lr)
    criterion = nn.CrossEntropyLoss()

    if not os.path.exists('../results'):
        os.mkdir('../results')
    
    train_loss, train_acc, eval_loss, eval_acc, test_loss, test_acc, model_ = train_model(model, dataloader_dict, batch_size, criterion, optimizer, num_epochs, lr, device, strategy_name)
    results_frame = pd.DataFrame()
    results_frame['train_loss'] = train_loss        
    results_frame['train_acc'] = train_acc        
    results_frame['eval_loss'] = eval_loss        
    results_frame['eval_acc'] = eval_acc
    results_frame.to_csv('../results/{}_training_results_{}_{}.csv'.format(strategy_name, test_loss, test_acc), index=False)
    
    if strategy_name != 'normal':
        print('...Beginning AL training...')        
        for active_learning_round in range(5):
            dataloader_dict = query_pool(model_, dataloader_dict,
                                         strategy_name, batch_size, query_size=20)
            train_loss, train_acc, eval_loss, eval_acc, test_loss, test_acc, model_ = train_model(model, dataloader_dict, batch_size, criterion, optimizer, num_epochs, lr, device, strategy_name)
            results_frame = pd.DataFrame()
            results_frame['train_loss'] = train_loss        
            results_frame['train_acc'] = train_acc        
            results_frame['eval_loss'] = eval_loss        
            results_frame['eval_acc'] = eval_acc
            results_frame.to_csv(
                '../results/{}_training_results_{}_{}_al_round_{}.csv'.format(strategy_name,
                test_loss, test_acc, active_learning_round + 1), index=False)

        # delete the model to free memory
        del model_
        torch.cuda.empty_cache()
    else:
        del model_
        torch.cuda.empty_cache()

if __name__ == '__main__':
    run_strategy('normal')
    run_strategy('max_entropy')
    run_strategy('mean_std')
    run_strategy('bald')