import kerastuner
import numpy as np
from sklearn import model_selection
from inception_hp import HyperInception
from inception_hp_coral import HyperInceptionCoral
from keras.utils import to_categorical
from argparse import ArgumentParser


class CVTuner(kerastuner.engine.tuner.Tuner):
    def run_trial(self, trial, x, y, batch_size=32, epochs=400):
        cv = model_selection.KFold(3, shuffle=True)
        val_losses = []
        val_acc = []

        # if batch_size is None:
        # batch_size = trial.hyperparameters.Int('batch_size', 16, 64,
        #                                        step=16)
        # if epochs is None:
        epochs = trial.hyperparameters.Int('epochs', 100, 450, 10)
        # epochs = 1
        #epochs = 5
        for train_indices, test_indices in cv.split(x):
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
            eval_metrics = model.evaluate(x_test, y_test)
            val_losses.append(eval_metrics[0])
            val_acc.append(eval_metrics[1])
            print(f'Validation loss: {eval_metrics[0]}')
            print(f'Validation accuracy: {eval_metrics[1]}')
            # print(f'Val loss: {val_losses[-1]}')

        print(f'Mean val loss: {np.mean(val_losses)}')
        print(f'Mean val acc: {np.mean(val_acc)} +- {np.std(val_acc)}')

        self.oracle.update_trial(
            trial.trial_id, {'val_loss': np.mean(val_losses)})
        self.save_model(trial.trial_id, model)


def main(args):
    dataset = np.load(args.dataset)
    indices = np.load(args.idx)
    x = dataset['mts']
    y = dataset['labels']
    x_train = x[indices['train_idx'], ...]
    y_train = y[indices['train_idx']]
    y_oh = to_categorical(y_train)
    #y_oh = y_train
    model = HyperInception(num_classes=3, input_shape=x_train.shape[1:])
    #model = HyperInceptionCoral(num_classes=3, input_shape=x_train.shape[1:])
    tuner = CVTuner(hypermodel=model,
                    oracle=kerastuner.oracles.BayesianOptimization(
                        objective='val_loss', max_trials=500))

    tuner.search(x_train, y_oh)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='/home/filipkr/Documents/xjob/data/datasets/data_Grazia-Deledda.npz')
    parser.add_argument('--idx', default='/home/filipkr/Documents/xjob/motion-analysis/classification/tsc/idx.npz')
    args = parser.parse_args()
    main(args)
