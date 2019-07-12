import argparse
from classifier import ImageClassifier
from dataprocessing import process_image
import json
import numpy as np
import torch


def main():
    """
    Set up arguments to be used for inference
    """
    parser = argparse.ArgumentParser(description='Trains a new network on a dataset and save the model as a checkpoint.')
    parser.add_argument('image_path',
                        type=str,
                        help='Image file')
    parser.add_argument('checkoint',
                        type=str,
                        help='Checkpoint file')
    parser.add_argument('--top_k',
                        dest='top_k',
                        metavar='T',
                        type=int,
                        default=1,
                        help='Top K most likely classes')
    parser.add_argument('--category_names',
                        dest='category_names',
                        metavar='C',
                        type=str,
                        default='cat_to_name.json',
                        help='Mapping of categories to real names')
    parser.add_argument('--gpu',
                        dest='gpu',
                        type=bool,
                        nargs='?',
                        default=False,
                        const=True,
                        help='Use GPU for inference')

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.gpu else 'cpu'

    # load category mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # load the model
    model = ImageClassifier(device)
    model.load(args.checkoint)

    # predict
    probs, names = model.predict(process_image(args.image_path), args.top_k, cat_to_name)

    for prob, name in zip(probs[0], names):
        print(f'{name}: {100 * np.exp(prob):.2f}%')


if __name__ == '__main__':
    main()

