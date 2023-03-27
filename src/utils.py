import torch
from torch.utils.data import Dataset, Subset, DataLoader, SubsetRandomSampler
import random
import numpy as np
from typing import Tuple, List, Dict, Any
from numpy import array
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
from tqdm import tqdm
from PIL import Image

def run_uncertainty(model, dataset: DataLoader, device: torch.device, uncertainty_rounds: int=20) -> array:
    # runs uncertainty on the sample 
    with torch.no_grad():
        for inputs, labels in dataset:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            predictions = np.stack([torch.softmax(model(inputs), dim=-1).cpu().numpy()
                for _ in tqdm(range(uncertainty_rounds), desc ="Running Uncertainty")])
        
        return predictions, labels

def compute_entropy(models_predictions: array) -> Tuple[array, array]:
    """
    Computes the entropy of the predictions (as measure of uncertainty)
    If `is_bald` we compute both the entropy and the average entropy
    """
    # average predictions
    models_predictions = models_predictions.mean(axis=0)
    entropy = np.sum(-models_predictions*np.log(models_predictions), -1)
    expectation_entropy = np.mean(entropy, 0)
    return entropy, expectation_entropy

def compute_max_entropy(entropies: array, top_k: int = 10) -> array:
    """
    Returns the datapoints of the current subset that maximizes the entropy
    """
    top_indices = (-entropies).argsort()[:top_k]
    return top_indices

def compute_mean_std(model_prediction: array, top_k: int = 10) -> Tuple[array, array]:
    """
    Computes the mean std, and select top_k data point that maximize it
    """
    std = model_prediction.std(-1)
    mean_std = std.mean(0)
    top_indices = (-mean_std).argsort()[:top_k]
    return mean_std, top_indices

def compute_bald(entropies: array, average_entropies: array, top_k: int = 10) -> Tuple[array, array]:
    """
    BALD maximizes the mutual information between the model's prediction and its posterior
    I[y,w|x,d_train] = H[y|x,d_train]-E_{p(w|d_train)}[H[y|x,w]]
    returns the information gain and the top_k data point which maximized it
    """
    mutual_information = entropies - average_entropies
    top_indices = (-mutual_information).argsort()[:top_k]
    return mutual_information, top_indices

class BasicCNN(nn.Module):
    def __init__(self, channels=3, n_filters=32, kernel_size=4, pooling_size=2,
                 num_classes=2, dense_size=128, image_size=224):
        super(BasicCNN, self).__init__()
        """
        n_filters: number of filters
        kernel_size: kernel filter size
        pooling: pooling size
        dropout: dropout rate
        num_classes: # of output classes
        hyper parameters chosen as stated in the paper
        """
        self.conv1 = nn.Conv2d(channels, n_filters, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.50)
        self.pooling = nn.MaxPool2d(pooling_size)
        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size)
        self.activation = nn.ReLU()
        self.dense = nn.Linear(n_filters * ((image_size - 2 * kernel_size + 2) // 2) * ((image_size - 2 * kernel_size + 2) // 2), dense_size)
        self.classifier = nn.Linear(dense_size, num_classes)

    def forward(self, x):
        # convolution-relu-convolution-relu-max pooling-dropout-dense-relu-dropout-dense-softmax
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout2(x)
        out = self.classifier(x)
        return out

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.imgs = data
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, i):
        image_path, label = self.imgs[i]
        image = Image.open(image_path)
        image = self.transform(image)
        return image, int(label)

def preprocessing(data_dir: str, batch_size: int, strategy_name: str) -> Dict:
    # Data augmentation and normalization for training & Just normalization for validation
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

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                      for x in ['train', 'val', 'test']}

    dataloaders_dict = {}     
    dataloaders_dict['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders_dict['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)

    if strategy_name != 'normal':
        # We begin by creating an initial training
        # set of 80 negative examples and 20 positive examples from our training data, as well as a pool set from the remaining data. 
        training_images = image_datasets['train'].imgs
        negative_images = [sample for sample in training_images if int(sample[1]) == 0] # samples that are non-cancerous
        positive_images = [sample for sample in training_images if int(sample[1]) == 1] # samples that are cancerous

        initial_training_negatives, pool_negatives = negative_images[:80], negative_images[80:]
        initial_training_positives, pool_positives = positive_images[:20], positive_images[20:]

        initial_training_samples = initial_training_negatives + initial_training_positives
        pool_samples = pool_negatives + pool_positives

        random.shuffle(pool_samples)

        dataloaders_dict['train'] = DataLoader(dataset=CustomDataset(initial_training_samples, data_transforms['train']), batch_size=batch_size,
                                                              shuffle=True, num_workers=4)
        dataloaders_dict['pool'] = DataLoader(dataset=CustomDataset(pool_samples, data_transforms['val']), batch_size=len(pool_samples),
                                                               shuffle=False, num_workers=4) # taking all in one
    else:
        dataloaders_dict['train'] = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=False, num_workers=4)

    print('Train: {}, Dev: {}, Test: {}'.format(len(dataloaders_dict['train'].dataset.imgs),
                                                            len(dataloaders_dict['val'].dataset.imgs), len(dataloaders_dict['test'].dataset.imgs)))
    return dataloaders_dict

# adapted from Pytorch Tutorial on Pretrained CV models
def train_model(model: BasicCNN, dataloaders: Dict,
                batch_size: int, criterion: torch.nn.CrossEntropyLoss,
                optimizer: torch.optim.Optimizer, num_epochs: int, lr: float,
                device: torch.device, strategy: str, query_size: int) -> Tuple[List, List, List, List, float, float, BasicCNN]:
    
    if not os.path.exists('../models'):
        os.mkdir('../models')

    best_loss = 1000
    train_loss, train_acc, eval_loss, eval_acc = [], [], [], []
    path = os.path.join('../models/isic_basic_cnn_{}_{}_{}_{}_{}.pt'.format(batch_size, num_epochs, lr, strategy, query_size))
    
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)            

        for phase in ['train', 'val']:            
            if phase == 'train':
                model.train()
            else:
                model.eval()
        
            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.item())
            else:
                eval_loss.append(epoch_loss)
                eval_acc.append(epoch_acc.item())

            print('{} loss: {:.4f} acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_loss < best_loss:
                # selecting the weights with lowest eval loss
                best_loss = epoch_loss
                torch.save(model.state_dict(), path)
    
    checkpoints = torch.load(path)
    model.load_state_dict(checkpoints)

    test_loss, test_acc = test_model(model, dataloaders, criterion, device)
    print('Best val loss: {:4f}'.format(best_loss))
    print('Test loss: {:4f}'.format(test_loss))
    print('Test Accuracy: {:4f}'.format(test_acc))
    
    return train_loss, train_acc, eval_loss, eval_acc, test_loss, test_acc, model

def test_model(model: BasicCNN, dataloaders: Dict,
               criterion: torch.nn.CrossEntropyLoss, 
               device: torch.device) -> Tuple[float, float]:
    loss = 0.0
    corrects = 0.0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
        loss += loss.item() * inputs.size(0)
        corrects += torch.sum(preds == labels.data)
    test_loss = loss / len(dataloaders['test'].dataset)
    test_acc = corrects.double() / len(dataloaders['test'].dataset)
    return test_loss, test_acc.item()
 