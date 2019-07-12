import argparse
import torch

from classifier import ImageClassifier
from dataprocessing import DataProcessor


def main():
    """
    Set up arguments to be used in the training
    """
    parser = argparse.ArgumentParser(description='Trains a new network on a dataset and save the model as a checkpoint.')
    parser.add_argument('data_dir',
                        type=str,
                        help='Dataset directory')
    parser.add_argument('--save_dir',
                        dest='save_dir',
                        metavar='S',
                        type=str,
                        help='Checkpoint directory')
    parser.add_argument('--arch',
                        dest='arch',
                        type=str,
                        default='vgg16',
                        choices=['resnet18', 'resnet34', 'alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16',
                                 'vgg16_bn', 'vgg19', 'vgg19_bn'],
                        help='Network architecture')
    parser.add_argument('--learning_rate',
                        dest='learning_rate',
                        metavar='LR',
                        type=float,
                        default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_units',
                        dest='hidden_units',
                        metavar='H',
                        type=int,
                        default=256,
                        help='List of hidden units neurons')
    parser.add_argument('--dropout',
                        dest='dropout',
                        metavar='D',
                        type=float,
                        default=0.2,
                        help='Dropout for the hidden layers (one less than the number of hidden units)')
    parser.add_argument('--epochs',
                        dest='epochs',
                        metavar='E',
                        type=int,
                        default=25,
                        help='Number of epochs to run the training for')
    parser.add_argument('--n_classes',
                        dest='n_classes',
                        metavar='N',
                        type=int,
                        default=102,
                        help='Number of classes to classify')
    parser.add_argument('--gpu',
                        dest='gpu',
                        type=bool,
                        nargs='?',
                        default=False,
                        const=True,
                        help='Use GPU for training')

    args = parser.parse_args()
    args.save_dir = args.save_dir if args.save_dir else args.data_dir
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.gpu else 'cpu'

    # create the model
    model = ImageClassifier(device)
    model.compile(args.arch, args.hiden_units, args.dropout, args.n_classes, args.learning_rate)

    # Load data
    dataloaders, image_datasets = DataProcessor(args.data_dir).create_loaders()

    model.train(dataloaders, args.epochs, image_datasets)
    model.save(f'{args.arch}_checkpoint.pth')


if __name__ == '__main__':
    main()
