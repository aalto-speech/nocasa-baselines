# nocasa-baselines
Baseline codes for the NOCASA Challenge

In this repository we share the baseline codes for the [IEEE-MLSP NOCASA Challenge](https://teflon.aalto.fi/nocasa-2025/)

## SVM model training

Steps to run the code:

1. Extract and copy or link the competition data (train and test folders and the train.csv and test.csv) from Zenodo in the same folder as the *.py files
2. Install the required python packeges (see requirements.txt)
3. Run the training script:

` python train_svm_baseline.py`

The script needs to be in the root folder of the data (i.e. where the csv files can be found)

During the first run, it will compute the COMPARE-16 features using OpenSmile and store them in separate csv files (train_compare_feats.csv and test_compare_feats.csv) 

The trained SVM model (sklearn pipeline of an StandardScaler, and a LinearSVC model) will be saved using pickle

The script will also evaluate the model and generate a prediction file (`test_predicted.csv`) compatible with the submission system

4. After training, the `eval_svm.py` can be used to create predictions on the test set using the model saved in a pickled file

The official pickled SVM baseline model can be found in this repo (see `svm_model.pkl`)

On the test set the model's performance:
| F1 | UAR | Accuracy | MAE (mean absolute error) |
| --- | --- | --- | --- |
| 22.22 | 22.14 | 32.74% | 1.05 |

## End-to-end wav2vec 2.0 training
TODO
