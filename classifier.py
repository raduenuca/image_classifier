import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler, SGD
from torchvision import models

import numpy as np

import io
import time
import os
import copy


class ImageClassifier():
    def __init__(self, device):
        """
        :param device: CPU or CUDA depending on the availability and choice when starting
        """
        self.device = device

    def compile(self, arch, hidden_units, dropout, n_classes, learning_rate):
        """
        Compiles the model
        :param arch: Chosen architecture
        :param hidden_units: Number of neurons in the hidden unit
        :param dropout: Dropout percentage for the hidden unit
        :param n_classes: Number of classes to predict
        :param learning_rate: Learning rate for the optimizer
        """
        self.model = getattr(models, arch)(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        if arch.startswith('resnet'):
            n_inputs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Linear(n_inputs, hidden_units),
                                          nn.ReLU(),
                                          nn.Dropout(dropout),
                                          nn.Linear(hidden_units, n_classes),
                                          nn.LogSoftmax(dim=1))
        elif arch.startswith('alexnet') or arch.startswith('vgg'):
            n_inputs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Sequential(nn.Linear(n_inputs, hidden_units),
                                                     nn.ReLU(),
                                                     nn.Dropout(dropout),
                                                     nn.Linear(hidden_units, n_classes),
                                                     nn.LogSoftmax(dim=1))
        else:
            print("Invalid model name, exiting...")
            exit()

        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        if arch.startswith('resnet'):
            self.optimizer = SGD(self.model.fc.parameters(), lr=self.learning_rate, momentum=0.9)
        elif arch.startswith('alexnet') or arch.startswith('vgg'):
            self.optimizer = SGD(self.model.classifier[6].parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            print("Invalid model name, exiting...")
            exit()

        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        print(self.model)


def train(self, dataloaders, num_epochs, image_datasets):
    """
    Trains the model on data
    :param dataloaders: Dataloaders for train, test and validate sets
    :param num_epochs: Number of epochs to train on
    :param image_datasets: ImageDatasets
    :return:
    """
    pass


def save(self, file_name):
    """
    Saves a model check point
    :param file_name: Path to where the model is saved
    :return:
    """
    pass
