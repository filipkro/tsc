import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
import seaborn


def main(args):
    path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/prob_matrices4.npz'
    path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/Mikhail-Sholokhov_prob_matrices.npz'
    path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/Olga-Tokarczuk_prob_matrices15.npz'
    if args.hists != '':
        path = args.hists
    data = np.load(path)
    # if '-test' in path:
    #     combs = np.reshape(data['comb'], (-1, 3, 3))
    print(data['reps'].shape)
    reps = np.reshape(data['reps'], (-1, 3, 3))
    combs = np.reshape(data['comb'], (-1, 3, 3))
    reps_max = np.reshape(data['reps_max'], (-1, 3, 3))
    combs_max = np.reshape(data['comb_max'], (-1, 3, 3))
    # probs = np.reshape(data['probs'], -1)

    print(reps.shape)
    print(combs.shape)


    seaborn.set_color_codes('muted')
    # print(np.reshape(reps, (-1,3,3)).shape)
    # idx = np.where(probs > -0.1)[0]
    # p = seaborn.histplot(data=probs[idx],bins=15,kde=False, binrange=(0,1))
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # p.set_ylabel('Count', fontsize=18)
    # p.set_xlabel('Probabilities', fontsize=18)
    # plt.ylabel(fontsize=16)
    # plt.show()
    labels = ['Good', 'Fair', 'Poor']
    for i in range(3):
        # if i > 1:
        #     break
        # bins = 5 if i == 2 else 10

        label = labels[i]

        idx_reps = np.where(reps[:, i, 0] > -0.1)[0]
        idx_combs = np.where(combs[:, i, 0] > -0.1)[0]
        # seaborn.histplot(data=reps[idx_reps,i,0],bins=15,kde=True, binrange=(0,1))

        # fig, axs = plt.subplots(3, gridspec_kw={'hspace':0.05})
        # fig.suptitle(f'Repetitions, correct label: {i}', fontsize=18)
        # # fig.tight_layout()
        # ymax = 0
        # for j in range(3):
        #     # axs[j].hist(reps[idx_reps, i, j], density=False,
        #     #             range=(0, 1), bins=15)
        #     col = 'b' if j == i else 'r'
        #     p = seaborn.histplot(data=reps[idx_reps,i,j],bins=15,kde=False, binrange=(0,1),  ax=axs[j], color=col)
        #     plt.sca(axs[j])
        #     xmin, xmax, ymin, ymax = plt.axis()
        #     if ymax == 1:
        #         plt.yticks([1],labels=['1'])
        #     else:
        #         plt.yticks(fontsize=12, rotation=60)
        #
        #     p.set_ylabel(f' Label {j}', fontsize=14)
        #     if j < 2:
        #         axs[j].axes.xaxis.set_ticklabels([])
        # plt.xticks(fontsize=15)
        # p.set_xlabel('Probabilities', fontsize=15)
        #
        #
        # # for j in range(3):
        # #     axs[j].set_ylim(0, ymax)
        #
        # fig, axs = plt.subplots(3, gridspec_kw={'hspace':0.05})
        # fig.suptitle(f'Predictions for repetitions, correct label: {i}', fontsize=18)
        #
        # ymax = 0
        # for j in range(3):
        #     col = 'b' if j == i else 'r'
        #     idx_max = np.where(reps_max[:, i, j] > -0.1)[0]
        #     p = seaborn.histplot(data=reps_max[idx_max, i, j],bins=15,kde=False, binrange=(0,1),  ax=axs[j], color=col)
        #     plt.sca(axs[j])
        #     xmin, xmax, ymin, ymax = plt.axis()
        #     if ymax == 1:
        #         plt.yticks([1],labels=['1'])
        #     else:
        #         plt.yticks(fontsize=12, rotation=60)
        #
        #     p.set_ylabel(f' Label {j}', fontsize=14)
        #
        #     if j < 2:
        #         axs[j].axes.xaxis.set_ticklabels([])
        # plt.xticks(fontsize=15)
        # p.set_xlabel('Probabilities', fontsize=15)
        #
        # fig, axs = plt.subplots(3, gridspec_kw={'hspace':0.05})
        # fig.suptitle(f'Combinations, correct label: {i}', fontsize=18)
        #
        # ymax = 0
        # for j in range(3):
        #     col = 'b' if j == i else 'r'
        #     p=seaborn.histplot(data=combs[idx_combs, i, j],bins=15,kde=False, binrange=(0,1),  ax=axs[j], color=col)
        #     plt.sca(axs[j])
        #     xmin, xmax, ymin, ymax = plt.axis()
        #     if ymax == 1:
        #         plt.yticks([1],labels=['1'])
        #     else:
        #         plt.yticks(fontsize=12, rotation=60)
        #     p.set_ylabel(f' Label {j}', fontsize=14)
        #     if j < 2:
        #         axs[j].axes.xaxis.set_ticklabels([])
        # plt.xticks(fontsize=15)
        # p.set_xlabel('Probabilities', fontsize=15)


        fig, axs = plt.subplots(3, gridspec_kw={'hspace':0.05})
        fig.suptitle(f'Correct label: {i}, {label}', fontsize=20)
        for j in range(3):
            col = 'b' if j == i else 'r'
            idx_max = np.where(combs_max[:, i, j] > -0.1)[0]
            p = seaborn.histplot(data=combs_max[idx_max, i, j],bins=15,kde=False, binrange=(0,1),  ax=axs[j], color=col)
            plt.sca(axs[j])
            xmin, xmax, ymin, ymax = plt.axis()
            if ymax == 1:
                plt.yticks([0.5, 1],labels=['1', ''])
                plt.yticks(fontsize=14, rotation=60)
            else:
                plt.yticks(fontsize=14, rotation=60)

            p.set_ylabel(f' Label {j}', fontsize=16)
            if j < 2:
                axs[j].axes.xaxis.set_ticklabels([])
        plt.xticks(fontsize=15)
        p.set_xlabel('Probabilities', fontsize=15)



    plt.show()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--hists', default='')
    parser.add_argument('--save', default='')
    main(parser.parse_args())
