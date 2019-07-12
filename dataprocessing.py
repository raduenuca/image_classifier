import io
import json
import os

import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image


class ImageLoader():
    def __init__(self, data_dir, cat_file='cat_to_name.json', train_batch_size=64, test_batch_size=32,
                 means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225], input_size=224, resize=256):
        """
        :param data_dir: The path to the directory that contains the data
        :param cat_file: The path to the file that contains the mapping to class names
        :param train_batch_size: The size of the training batch
        :param test_batch_size: The size of the testing and validation batch
        :param means: The means used to normalize the images
        :param stds: The standard deviations used to normalize the images
        :param input_size: The input image size based on network architecture
        :param resize: The size of the image scaling
        """
        self.data_dir = data_dir
        self.batch_sizes = {'train': train_batch_size, 'valid': test_batch_size, 'test': test_batch_size}
        self.test_batch_size = test_batch_size
        self.means = means
        self.stds = stds
        self.input_size = input_size
        self.resize = resize

        with open('cat_to_name.json', 'r') as f:
            self.cat_to_name = json.load(f)

    def create_loaders(self):
        """
        Create the loaders
        :return: Dataloaders and datasets
        """
        data_transforms = {
            'train': transforms.Compose([
                # transforms.Resize(scale),
                transforms.RandomRotation(degrees=30),
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds)
            ]),
            'valid': transforms.Compose([
                transforms.Resize(self.resize),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds)
            ]),
            'test': transforms.Compose([
                transforms.Resize(self.resize),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds)
            ])
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), data_transforms[x]) for x in
                          ['train', 'valid', 'test']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=self.batch_sizes[x]) for x in
                       ['train', 'valid', 'test']}

        return dataloaders, image_datasets


