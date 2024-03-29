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
def create_train_val_splits(path: str) -> None:
    # Splits the training set into train and validation
    dataset = pd.read_csv(path)
    label_map = {'benign': 0, 'malignant': 1}
    dataset['labels'] = dataset['labels'].apply(lambda x: label_map[x])
    train, val = train_test_split(dataset, stratify=dataset['labels'], train_size=0.78, random_state=1234) # attempt to get 700/200 for train and val

    train.to_csv('../train_split.csv', index=False)
    val.to_csv('../val_split.csv', index=False)


# train_path = '../ISBI2016_Training_Labels.csv'
# create_train_val_splits(train_path)

def make_dirs() -> None:
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

def create_images_to_dir(dataset_split: str, dataset_path: str) -> None:
    data_paths, data_labels = get_image_and_labels(dataset_path)
    directory = os.path.join('../', 'data', dataset_split)
    if dataset_split in ['train', 'val']:
        base_path = '../ISBI2016_Training_Data'
    else:
        base_path = '../ISBI2016_Test_Data'
    total_data = len(data_paths)
    for index in tqdm(range(total_data), desc="Data Creation Progress"):
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


input_size = (224, 224)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}


def query_pool(model: BasicCNN, device: torch.device, dataloaders: Dict, strategy: str,
               batch_size: int, query_size: int, reverse: bool = False) -> Dict:
    # set the model's dropout unit in training mode
    set_dropout(model)
    # run the model on the pool
    predictions, labels = run_uncertainty(model, dataloaders['pool'], device)
    entropy, expectation_entropy = compute_entropy(predictions)
    if strategy == 'max_entropy':
        top_indices = compute_max_entropy(entropy, query_size, reverse)
    if strategy == 'mean_std':
        mean_std, top_indices = compute_mean_std(predictions, query_size, reverse)
    if strategy == 'bald':
        mutual_information, top_indices = compute_bald(entropy, expectation_entropy, query_size, reverse)

    top_indices = top_indices.tolist()
    training_images = dataloaders['train'].dataset.imgs
    pool_images = dataloaders['pool'].dataset.imgs

    print('Initial size: Train = {}, Pool = {}'.format(len(training_images), len(pool_images)))

    # add indices to the training set
    for index in top_indices:
        training_images.append(pool_images[index])

    # removes indices from the pool set
    new_pool_images = []

    for _ in range(len(pool_images)):
        if _ not in top_indices:
            new_pool_images.append(pool_images[_])

    # re-asigns the updated data loaders with respective data transformation
    new_dataloaders = {}
    new_dataloaders['val'] = dataloaders['val']
    new_dataloaders['test'] = dataloaders['test']

    del dataloaders
    del pool_images

    new_dataloaders['train'] = DataLoader(CustomDataset(training_images, data_transforms['train']),
                                          batch_size=batch_size, shuffle=True)
    new_dataloaders['pool'] = DataLoader(CustomDataset(new_pool_images, data_transforms['val']),
                                         batch_size=len(new_pool_images), shuffle=False)

    print('Size after Update: Train = {}, Pool = {}'.format(len(new_dataloaders['train'].dataset.imgs),
                                                            len(new_dataloaders['pool'].dataset.imgs)))

    print('Train: {}, Dev: {}, Test: {}'.format(len(new_dataloaders['train'].dataset.imgs),
                                                len(new_dataloaders['val'].dataset.imgs),
                                                len(new_dataloaders['test'].dataset.imgs)))

    return new_dataloaders


def run_strategy(strategy_name: str, query_size: int, reverse: bool = False, time='train',
                 model_list: List = None) -> None:
    
    print('Running with strategy == {}'.format(strategy_name))
    data_dir = '../data'
    batch_size, num_epochs, lr = 8, 100, 1e-4

    dataloader_dict = preprocessing(data_dir, batch_size, strategy_name)

    training_data_points = len(dataloader_dict['train'])

    # from the paper: weight_decay =  (1 - p)lsquared/N where N = |training_set|, p = 0.5, and l_squared = 0.5
    p, l_squared = 0.5, 0.5
    weight_decay = ((1-p)*l_squared)/training_data_points
    criterion = nn.CrossEntropyLoss()

    if not os.path.exists('../results'):
        os.mkdir('../results')

    if not os.path.exists('../results/{}'.format(strategy_name)):
        os.mkdir('../results/{}'.format(strategy_name))
    
    if time == 'train':
        model = build_model()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if strategy_name != 'normal':
            # An acquisition function is then used to select the `100` most informative images from the pool set.
            print('...Training AL with strategy == {} and query_size == {}'.format(strategy_name,
                                                                                                    query_size))
            for active_learning_round in range(5):
                
                if not os.path.exists('../results/{}/Round_{}'.format(strategy_name, active_learning_round + 1)):
                    os.mkdir('../results/{}/Round_{}'.format(strategy_name, active_learning_round + 1))

                train_loss, train_acc, eval_loss, eval_acc, test_loss, test_acc, model_ = train_model(model,
                                                                                                    dataloader_dict,
                                                                                                    batch_size,
                                                                                                    criterion,
                                                                                                    optimizer,
                                                                                                    num_epochs,
                                                                                                    lr, device,
                                                                                                    strategy_name,
                                                                                                    query_size, reverse)

                results_frame = pd.DataFrame()
                results_frame['train_loss'] = train_loss
                results_frame['train_acc'] = train_acc
                results_frame['eval_loss'] = eval_loss
                results_frame['eval_acc'] = eval_acc
                results_frame.to_csv('../results/{}/Round_{}/training_results_{}_{}_query_{}_{}.csv'.format(strategy_name,
                                                                                        active_learning_round + 1,
                                                                                        test_loss, test_acc,
                                                                                        query_size, reverse), index=False)

                dataloader_dict = query_pool(model_, device, dataloader_dict,
                                                strategy_name, batch_size, query_size, reverse)

                # delete the model to free memory
                del model_
                del model
                torch.cuda.empty_cache()
                # build the model from scratch for new training round
                model = build_model()

                training_data_points = len(dataloader_dict['train'])
                p, l_squared = 0.5, 0.5
                # updated weight decay for the new training set and model for next active learning round
                weight_decay = ((1-p)*l_squared)/training_data_points
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        else:

            train_loss, train_acc, eval_loss, eval_acc, test_loss, test_acc, model_ = train_model(model, dataloader_dict,
                                                                                            batch_size, criterion,
                                                                                            optimizer,
                                                                                            num_epochs, lr, device,
                                                                                            strategy_name, query_size, reverse)
            results_frame = pd.DataFrame()
            results_frame['train_loss'] = train_loss
            results_frame['train_acc'] = train_acc
            results_frame['eval_loss'] = eval_loss
            results_frame['eval_acc'] = eval_acc
            results_frame.to_csv('../results/{}/training_results_{}_{}_{}.csv'.format(strategy_name, test_loss, test_acc, reverse),
                                index=False)

            del model_
            del model
            torch.cuda.empty_cache()
    else:
        for model_path, description in model_list:
            model = build_model()
            checkpoints = torch.load(model_path)
            model.load_state_dict(checkpoints)
            _1, _2 = test_model(model, dataloader_dict, criterion, device, 'test', description)
            del model
            torch.cuda.empty_cache()

if __name__ == '__main__':
    run_strategy('normal', 0)
    query_size_options = [115, 100, 90, 80, 70, 60, 50]
    # run with different query size to leverage impact of batch acquisition effect
    for query_size in query_size_options:
        run_strategy('max_entropy', query_size)
        run_strategy('mean_std', query_size)
        run_strategy('bald', query_size)
    
    # running baselines (with parameters stated in the paper) with `reverse` option which will take the least uncertain samples
    run_strategy('max_entropy', 100, reverse=True)
    run_strategy('mean_std', 100, reverse=True)
    run_strategy('bald', 100, reverse=True)

    # testing time to build confusion matrix maps
    model_list = [('/home/mila/b/bonaventure.dossou/ece526_course_project/models/isic_basic_cnn_8_100_0.0001_bald_100.pt', 'Bald'),
                  ('/home/mila/b/bonaventure.dossou/ece526_course_project/models/isic_basic_cnn_8_100_0.0001_max_entropy_70.pt', 'Max Entropy'),
                  ('/home/mila/b/bonaventure.dossou/ece526_course_project/models/isic_basic_cnn_8_100_0.0001_mean_std_100.pt', 'Mean Std')]

    run_strategy('', 0, time = 'test', model_list=model_list)