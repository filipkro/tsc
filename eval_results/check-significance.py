import numpy as np
from argparse import ArgumentParser
import scipy.stats as stats

def main(args):
    data = np.load(args.data)

    acc_comb = np.array(data['comb_acc'])
    acc_th = np.array(data['thresh_acc'])
    f1_comb = np.array(data['comb_f1'])
    f1_th = np.array(data['thresh_f1'])
    acc_rep = np.array(data['rep_acc'])
    f1_rep = np.array(data['rep_f1'])
    acc_ind = np.mean(np.array(data['acc_ind']),axis=1)
    f1_ind = np.mean(np.array(data['f1_ind']),axis=1)
    acc_ind0 = np.array(data['acc_ind'])[:,0]
    f1_ind0 = np.array(data['f1_ind'])[:,0]
    acc_ind1 = np.array(data['acc_ind'])[:,1]
    f1_ind1 = np.array(data['f1_ind'])[:,1]

    diff_acc_ind = acc_rep - acc_ind
    diff_f1_ind = f1_rep - f1_ind


    da0 = acc_rep - acc_ind0
    df0= f1_rep - f1_ind0
    da1 = acc_rep - acc_ind1
    df1= f1_rep - f1_ind1
    print(np.mean(da0))
    print(np.mean(df0))
    print(np.mean(da1))
    print(np.mean(df1))

    # print(f'Individual Acc m0:: {np.mean(da0) - stats.norm.ppf(0.6) * np.std(da0)/np.sqrt(10)}')
    # print(f'Individual F1 m0:: {np.mean(df0) - stats.norm.ppf(0.75) * np.std(df0)/np.sqrt(10)}')
    # print(f'Individual Acc m1:: {np.mean(da1) - stats.norm.ppf(0.51) * np.std(da1)/np.sqrt(10)}')
    # print(f'Individual F1 m1:: {np.mean(df1) - stats.norm.ppf(0.8) * np.std(df1)/np.sqrt(10)}')

    # print(f'Individual Acc:: {np.mean(diff_acc_ind) - stats.norm.ppf(0.99) * np.std(diff_acc_ind)/np.sqrt(10)}')
    # print(f'Individual F1:: {np.mean(diff_f1_ind) - stats.norm.ppf(0.99) * np.std(diff_f1_ind)/np.sqrt(10)}')

    acc_rep_diff = acc_comb - acc_rep
    f1_rep_diff = f1_comb - f1_rep

    print(f'Combination Acc:: {np.mean(acc_rep_diff) - stats.norm.ppf(0.99) * np.std(acc_rep_diff)/np.sqrt(10)}')
    print(f'Combination F1:: {np.mean(f1_rep_diff) - stats.norm.ppf(0.99) * np.std(f1_rep_diff)/np.sqrt(10)}')

    acc_diff = acc_th - acc_comb
    f1_diff = f1_th - f1_comb
    print(np.mean(acc_diff))
    print(np.mean(f1_diff))
    print(f'Threshold Acc:: {np.mean(acc_diff) - stats.norm.ppf(0.95) * np.std(acc_diff)/np.sqrt(10)}')
    print(f'Threshold F1:: {np.mean(f1_diff) - stats.norm.ppf(0.95) * np.std(f1_diff)/np.sqrt(10)}')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data')
    main(parser.parse_args())
