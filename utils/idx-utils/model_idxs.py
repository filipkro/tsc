import numpy as np

path = '/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz'

indices = np.load(path)
train_idx = indices['train_idx']
val_idx = np.random.choice(train_idx, 40, replace=False)

train_idx = [e for e in train_idx if e not in val_idx]

np.savez(path.split('.')[0] + '2.npz', test_idx=indices['test_idx'],
         test_subj=indices['test_subj'], train_idx=train_idx, val_idx=val_idx)

# print(indices.files)
