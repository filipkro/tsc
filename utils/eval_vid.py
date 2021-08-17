import numpy as np
from argparse import ArgumentParser, ArgumentTypeError
import os
import tensorflow.keras as keras

import coral_ordinal as coral
from confusion_utils import ConfusionCrossEntropy\


def main(args, datasets=None, datasets100=None):

    poes = ['femval', 'trunk', 'hip', 'kmfp']
    for poe in poes:
        model_path = '/home/filipkr/Documents/xjob/training/ensembles'
        if datasets is None or datasets100 is None:
            x = np.load(os.path.join(args.root, poe + '.npy'))
            x100 = np.load(os.path.join(args.root, poe + '-100.npy'))
        else:
            x = datasets[poe]
            x100 = datasets100[poe]

        if poe == 'femval':
            lit = 'Olga-Tokarczuk'
            models = ['coral-100-7000', 'xx-coral-100-7000', 'xx-conf-100-7000',
                      'xx-conf-100-11000', 'xx-conf-9000']
            weights = np.array([[1/3, 1.25/3, 1/3], [1/3, 1.25/3, 1/3],
                                [1/3, 0, 0], [0, 0, 1/3], [0, 1.25/3, 0]])
        elif poe == 'hip':
            lit = 'Sigrid-Undset'
            models = ['coral-100-13000', 'coral-13000', 'conf-100-10000',
                      'conf-10001', 'xx-coral-100-10003']
            weights = np.array([[1/4, 1.05/4, 1.5/4], [1/4, 1.05/4, 1.5/4],
                                [1/2, 0, 0], [0, 1.05/2, 0], [0, 0, 1.5/2]])
        elif poe == 'kmfp':
            lit = 'Mikhail-Sholokhov'
            models = ['inception-3010', 'xx-inception-3010', 'xx-conf-3010',
                      'conf-100-13000', 'xx-conf-3025']
            weights = np.array([[1/3, 1.25*1/3, 1.25/3], [1/3, 1.25*1/3, 1.25/3],
                                [1/3, 0, 0], [0, 1.25*1/3, 0], [0, 0, 1.25/3]])
        elif poe == 'trunk':
            lit = 'Isaac-Bashevis-Singer'
            models = ['coral-100-11', 'coral-100-10', 'xx-conf-100-11', 'conf-15',
                      'xx-coral-100-10']
            weights = np.array([[1/3, 1.15/3, 1/3], [1/3, 1.15/3, 1/3],
                                [1/3, 0, 0], [0, 1.15/3, 0],[0, 0, 1/3]])

        model_path = os.path.join(model_path, lit + '10')
        ensembles = [os.path.join(model_path, i) for i in models]
        paths = [os.path.join(root, 'model_fold_1.hdf5') for root in ensembles]
        all_probs = np.zeros((len(models), x.shape[0], 3))

        for model_i, model_path in enumerate(paths):

            input = x100 if '-100-' in model_path else x
            model = keras.models.load_model(model_path, custom_objects={
                                            'CoralOrdinal': coral.CoralOrdinal,
                                            'OrdinalCrossEntropy':
                                            coral.OrdinalCrossEntropy,
                                            'MeanAbsoluteErrorLabels':
                                            coral.MeanAbsoluteErrorLabels,
                                            'ConfusionCrossEntropy':
                                            ConfusionCrossEntropy})

            print(input.shape)
            print(lit)
            print(poe)
            print(model_path)
            result = model.predict(input)
            probs = coral.ordinal_softmax(
                result).numpy() if 'coral' in model_path else result

            probs = probs * weights[model_i, ...]

            all_probs[model_i, ...] = probs

        ensemble_probs = np.sum(all_probs, axis=0)
        # threshold
        ensemble_probs = (ensemble_probs > 0.2) * ensemble_probs
        # ev fel shape ....
        summed = np.mean(ensemble_probs, axis=0)
        pred_combined = int(np.argmax(np.mean(ensemble_probs, axis=0)))
        # pred = np.argmax(pred_subj, axis=1)

        print(f'Prediction for POE, {poe}: {pred_combined}')
        print(f'Certainties: {np.mean(ensemble_probs, axis=0)}')
        print(f'Summed-score: {summed}')

        print(ensemble_probs)



def str2bool(v):
    ''' pass bool wtih argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root', default='')
    # parser.add_argument('data')
    args = parser.parse_args()
    main(args)
