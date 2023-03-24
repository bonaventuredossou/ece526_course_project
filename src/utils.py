import torch
from torch.utils.data import Dataset, Subset, DataLoader
import random
import numpy as np
from typing import Tuple, List, Dict
from numpy import array
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import os

def sample_subset(augmentation_set: Dataset) -> Tuple[DataLoader, list]:
    """
    This function samples a mini-batch from the augmenting set (referred to as pool in the original paper)
    augmentation_set: Pytorch Dataset object
    returns a dataloader ready to be ran on the model for uncertainty quantification, and indices of
    selected samples
    """
    selected_indices = random.sample(range(0, len(augmentation_set)), k=500) # selects randomly 500 indices
    current_subset = Subset(augmentation_set, selected_indices)
    return DataLoader(current_subset, shuffle=False, batch_size=len(current_subset)), selected_indices

def run_uncertainty(model, dataset: DataLoader, uncertainty_rounds: int=20) -> array:
    # runs uncertainty on the sample 
    with torch.no_grad:
        image, labels = dataset
        predictions = np.stack([torch.softmax(model(image), dim=-1).cpu().numpy()
                                for _ in range(uncertainty_rounds)])
        return predictions

def compute_entropy(model_predictions: array, is_bald: bool) -> Tuple[array, float]:
    """
    Computes the entropy of the predictions (as measure of uncertainty)
    If `is_bald` we compute both the entropy and the average entropy
    """
    # average predictions
    models_predictions = models_predictions.mean(axis=0)
    entropy = np.sum(-models_predictions*np.log(models_predictions), -1)
    average_entropy = np.mean(entropy, 0)
    if is_bald:
        return entropy, average_entropy
    else:
        return entropy

def compute_max_entropy(entropies: array, top_k: int = 100) -> array:
    """
    Returns the datapoints of the current subset that maximizes the entropy
    """
    top_indices = (-entropies).argsort()[:top_k]
    return top_indices

def compute_mean_std(model_prediction: array, top_k: int = 100) -> Tuple[array, array]:
    """
    Computes the mean std, and select top_k data point that maximize it
    """
    std = model_prediction.std(-1)
    mean_std = std.mean(0)
    top_indices = (-mean_std).argsort()[:top_k]
    return mean_std, top_indices

def compute_bald(entropies: array, average_entropies: array, top_k: int = 100) -> Tuple[array, array]:
    """
    BALD maximizes the mutual information between the model's prediction and its posterior
    I[y,w|x,d_train] = H[y|x,d_train]-E_{p(w|d_train)}[H[y|x,w]]
    returns the information gain and the top_k data point which maximized it
    """
    mutual_information = entropies - average_entropies
    top_indices = (-mutual_information).argsort()[:top_k]
    return mutual_information, top_indices

# conv_kernel, kernel_size, pooling, dense layer, dropout
class BasicCNN(nn.Module):
    def __init__(self, channels=3, n_filters=32, kernel_size=4, pooling_size=2,
                 dropout=0.3, num_classes=2):
        super(BasicCNN, self).__init__()
        """
        n_filters: number of filters
        kernel_size: kernel filter size
        pooling: pooling size
        dropout: dropout rate
        num_classes: # of output classes
        """
        self.conv1 = nn.Conv2d(channels, n_filters, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.pooling = nn.MaxPool2d(pooling_size)
        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(n_filters, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(self.bn(x))
        x = self.conv2(x)
        x = self.activation(self.bn(x))
        x = self.conv3(x)
        x = self.activation(self.bn(x))
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        out = self.classifier(x)
        return out

def preprocessing(data_dir: str, batch_size: int) -> Dict[DataLoader]:
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

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    dataloaders_dict = {x[0]: torch.utils.data.DataLoader(image_datasets[x[0]], batch_size=batch_size, shuffle=x[1], num_workers=4) for x in [('train', True), ('val', False), ('test', False)]}
    return dataloaders_dict