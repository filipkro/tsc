import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
import seaborn

def main(args):
    femval = os.path.join(args.root, 'data_Olga-Tokarczuk.npz')
    trunk = os.path.join(args.root, 'data_Nadine-Gordimer.npz')
    pelvis = os.path.join(args.root, 'data_Sigrid-Undset.npz')
    kmfp = os.path.join(args.root, 'data_Mikhail-Sholokhov.npz')

    idx_root = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/'
    idx_femval = np.load(idx_root + 'idx-femval.npz')['test_idx']
    idx_trunk = np.load(idx_root + 'idx-trunk.npz')['test_idx']
    idx_pelvis = np.load(idx_root + 'idx-hip.npz')['test_idx']
    idx_kmfp = np.load(idx_root + 'idx-KMFP.npz')['test_idx']

    # femval = np.load(femval)['labels']
    # trunk = np.load(trunk)['labels']
    # pelvis = np.load(pelvis)['labels']
    # kmfp = np.load(kmfp)['labels']
    # ymax =430

    femval = np.load(femval)['labels'][idx_femval]
    trunk = np.load(trunk)['labels'][idx_trunk]
    pelvis = np.load(pelvis)['labels'][idx_pelvis]
    kmfp = np.load(kmfp)['labels'][idx_kmfp]
    ymax = 90

    for i in range(3):
        print(i)
        print(f'femval ratio: {np.sum(femval==i)/len(femval)}')
        print(f'trunk ratio: {np.sum(trunk==i)/len(trunk)}')
        print(f'pelvis ratio: {np.sum(pelvis==i)/len(pelvis)}')
        print(f'kmfp ratio: {np.sum(kmfp==i)/len(kmfp)}')

    # plt.hist(femval, bins=3)
    # seaborn.histplot(data=femval,discrete=True, bins=3)
    # plt.title('Femoral Valgus', fontsize=20)
    # plt.ylim(0,ymax)
    # # plt.xticks(ticks=[1/3, 1, 5/3], labels=['0', '1', '2'], fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.xticks(fontsize=16)
    # plt.figure()
    # plt.hist(trunk, bins=3)
    # plt.ylim(0,ymax)
    # plt.xticks(ticks=[1/3, 1, 5/3], labels=['0', '1', '2'], fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.title('Trunk', fontsize=20)
    # plt.figure()
    # plt.hist(pelvis, bins=3)
    # plt.ylim(0,ymax)
    # plt.xticks(ticks=[1/3, 1, 5/3], labels=['0', '1', '2'], fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.title('Pelvis', fontsize=20)
    # plt.figure()
    # plt.hist(kmfp,bins=3)
    # plt.ylim(0,ymax)
    # plt.xticks(ticks=[1/3, 1, 5/3], labels=['0', '1', '2'], fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.title('KMFP', fontsize=20)

    p = seaborn.histplot(data=femval,discrete=True, bins=3)
    plt.title('Femoral Valgus', fontsize=28)
    plt.ylim(0,ymax)
    plt.xticks(ticks=[0, 1, 2], labels=['0', '1', '2'], fontsize=20)
    plt.yticks(fontsize=20)
    p.set_ylabel('Count', fontsize=24)
    plt.legend([],[], frameon=False)
    plt.subplots_adjust(left=0.164,right=0.939)
    # plt.xticks(fontsize=16)
    plt.figure()
    p = seaborn.histplot(data=trunk,discrete=True, bins=3)
    plt.title('Trunk', fontsize=28)
    plt.ylim(0,ymax)
    plt.xticks(ticks=[0, 1, 2], labels=['0', '1', '2'], fontsize=20)
    plt.yticks(fontsize=20)
    p.set_ylabel('Count', fontsize=24)
    plt.legend([],[], frameon=False)
    plt.subplots_adjust(left=0.164,right=0.939)
    plt.figure()
    p = seaborn.histplot(data=pelvis,discrete=True, bins=3)
    plt.title('Pelvis', fontsize=28)
    plt.ylim(0,ymax)
    plt.xticks(ticks=[0, 1, 2], labels=['0', '1', '2'], fontsize=20)
    plt.yticks(fontsize=20)
    p.set_ylabel('Count', fontsize=24)
    plt.legend([],[], frameon=False)
    plt.subplots_adjust(left=0.164,right=0.939)
    plt.figure()
    p = seaborn.histplot(data=kmfp,discrete=True, bins=3)
    plt.title('KMFP', fontsize=28)
    plt.ylim(0,ymax)
    plt.xticks(ticks=[0, 1, 2], labels=['0', '1', '2'], fontsize=20)
    plt.yticks(fontsize=20)
    p.set_ylabel('Count', fontsize=24)
    plt.legend([],[], frameon=False)
    plt.subplots_adjust(left=0.164,right=0.939)


    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root')
    main(parser.parse_args())
