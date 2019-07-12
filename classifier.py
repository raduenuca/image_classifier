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
        :return:
        """
        pass

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

