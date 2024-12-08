I trained/evaluated the models using 10-fold cross validation along with a separate test set. The training is done in [`cross-val-train-classifier`](cross-val-train-classifier.py). I ended up using ensembles of different architectures, see thesis for more details. Hence I trained a number of different models for each dataset and ended up using models performing well on cross validation either overall or achieving good precision. Again, `cross-val-train-classifier.py` is a bit messy and is dependent on how the datasets are created/organised (seee motion analysis repo). I will make sure it matches with new code generating datasets, after doing this I will also provide some more instructions.

How models then are evaluated depends again on data and models used. The most recent code for this can be found in [`eval_consensus_cuts.py`](utils/eval_consesnsus_cuts.py). This code is for data augmented by cutting in the beginning and end of the sequences. It will run all the different models for the different cuts and classify as the likeliest class for the summed predicted probabilities from the ensembles. The different models in the ensembles are also weighted differently depending on what they are trained to recognise. 

The following command will run the cross validation training
```
python cross-val-train-classifier.py <save directory> <path to dataset>.npz <model type>
```
Model type is specified as e.g. `inception-coral` (the very long if statement in `cross-val-train-classifier.py`). The model used should be specified in a model file in the `classifiers` directory.

I think I would recommend rewriting this in PyTorch instead which I think is more intuitive, more useful to know, and what's used by `mmpose` for the keypoint extraction
