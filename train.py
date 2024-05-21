# Wylea's Part

# Source
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
# https://www.tensorflow.org/guide/basic_training_loops

# Relevant project links:
# https://github.com/232525/PureT/tree/main
# https://github.com/berniwal/swin-transformer-pytorch

# Imports
import itertools
import random
import numpy as np
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

# Set up logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Determine if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FullTrainingDataset class
class FullTrainingDataset(Dataset):
    def __init__(self, full_ds, offset, length):
        self.full_ds = full_ds
        self.offset = offset
        self.length = length
        assert len(full_ds) >= offset + length, "Parent Dataset not long enough"
        super(FullTrainingDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.full_ds[i + self.offset]

def trainTestSplit(dataset, val_share=0.2):
    val_offset = int(len(dataset) * (1 - val_share))
    return FullTrainingDataset(dataset, 0, val_offset), FullTrainingDataset(dataset, val_offset, len(dataset) - val_offset)

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def matplot_print(training_loader, classes):
    dataiter = iter(training_loader)
    images, labels = next(dataiter)

    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)
    print('  '.join(classes[labels[j]] for j in range(4)))

def main():
    # Define transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, assuming data is in 'train/' and 'val/' directories
    train_dir = './train'
    val_dir = './val'
    training_set = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    validation_set = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)

    # Create data loaders for our datasets; shuffle for training, not for validation
    train_ds, val_ds = trainTestSplit(training_set)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    # Report split sizes
    print(f'Training set has {len(training_set)} instances')
    print(f'Validation set has {len(validation_set)} instances')

    # Visualization
    classes = training_set.classes  # Get class names from the dataset
    matplot_print(train_loader, classes)

if __name__ == "__main__":
    main()
