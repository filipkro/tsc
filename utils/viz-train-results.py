import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# cols: loss,accuracy,val_loss,val_accuracy,lr

def main(args):
    data = np.genfromtxt(args.file, delimiter=',')[1:,:]
    # print(data)

    plt.plot(data[:,0], label='Training loss')
    plt.plot(data[:,2], label='Validation loss')
    plt.legend()

    plt.figure()
    plt.plot(data[:,1], label='Training accuracy')
    plt.plot(data[:,3], label='Validation accuracy')
    plt.legend()

    plt.figure()
    plt.plot(data[:,4])
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()
    main(args)
