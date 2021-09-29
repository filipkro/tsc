import numpy as np

poes = ['trunk', 'hip', 'femval', 'KMFP']
lits = {'hip': 'Sigrid-Undset', 'trunk': 'Isaac-Bashevis-Singer',
        'femval': 'Olga-Tokarczuk', 'KMFP': 'Mikhail-Sholokhov'}
path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx-'
dpath = '/home/filipkr/Documents/xjob/data/datasets/data_'

for poe in poes:
    orig = np.load(path + poe + '.npz')
    test = orig['test_idx']
    labels = np.load(dpath + lits[poe] + '.npz')['labels']
    # print(len(test))
    # print(len(set(test)))
    # print(type(list(test)))
    train = [e for e in range(len(labels)) if e not in test]
    # print(len(train))
    # print(len(set(train)))
    # print(type(train))
    # print(len(set(np.append(train, test))))

    if len(test) + len(train) == len(set(np.append(train, test))):
        np.savez(path + poe + '2.npz', test_idx=orig['test_idx'],
                 test_subj=orig['test_subj'], train_idx=train)

    # print(len(set(train.append(list(test)))))

    # print(len(set((list(test), train])))
    # print(e)
