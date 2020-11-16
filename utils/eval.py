import numpy as np
import sklearn
from argparse import ArgumentParser

def main(args):
    dataset = np.load(args.dataset)
    train_idx = np.load(args.train)
    x = dataset['mts']
    y = dataset['labels']
    x = np.delete(x, train_idx, axis=0)
    y = np.delete(y, train_idx)

    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('train')
    parser.add_argument('classifier')
    parser.add_argument('root')
    args = parser.parse_args()
    main(args)
