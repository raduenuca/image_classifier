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
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        self.epochs = num_epochs
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.scheduler.step()
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        if type(outputs) == tuple:
                            outputs, _ = outputs

                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(image_datasets[phase])

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.model.class_to_idx = image_datasets['train'].class_to_idx

    def evaluate(self, dataloader, dataset_size):
        """
        Evaluates the model against a test dataset
        :param dataloader: Dataloader for the test data
        :param dataset_size: Size of the dataset
        """
        since = time.time()

        avg_acc = 0
        acc_test = 0

        self.model.eval()  # Set model to evaluate mode

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)

                acc_test += torch.sum(preds == labels.data).item()

        avg_acc = acc_test / dataset_size

        elapsed_time = time.time() - since
        print()
        print(f'Evaluation completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')
        print(f'Avg acc (test): {avg_acc:.4f}')
        print('-' * 10)

    def save(self, file_name):
        """
        Saves a model check point
        :param file_name: Path to where the model is saved
        """
        checkpoint = {
            'model': self.model,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer_ft.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'class_to_idx': self.model.class_to_idx,
            'epochs' : self.epochs,
            'LR': self.learning_rate
        }

        torch.save(checkpoint, file_name)

    def load(self, filename):
        """
        Loads the model from a checkpoint file
        :param filename: Path to the checkpoint file
        """

        start_epoch = 0
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            start_epoch = checkpoint['epochs']

            self.model = checkpoint['model']
            self.model = self.model.to(self.device)
            self.model.class_to_idx = checkpoint['class_to_idx']
            self.model.load_state_dict(checkpoint['state_dict'])

            self.optimizer = SGD(self.model.classifier.parameters(), lr=(checkpoint['LR'] if 'LR' in checkpoint else 0.001), momentum=0.9)
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"=> loaded checkpoint '{filename}' (epoch {checkpoint['epochs']})")
        else:
            print(f"=> no checkpoint found at '{filename}'")