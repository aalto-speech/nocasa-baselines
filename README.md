# nocasa-baselines
Baseline codes for the NOCASA Challenge

In this repository, we share the baseline codes for the [IEEE-MLSP NOCASA Challenge](https://teflon.aalto.fi/nocasa-2025/)

## SVM model training

Steps to run the code:

1. Extract and copy or link the competition data (train and test folders and the train.csv and test.csv) from Zenodo in the same folder as the *.py files
2. Install the required python packages (see requirements.txt)
3. Run the training script:

` python train_svm_baseline.py`

The script needs to be in the root folder of the data (i.e. where the csv files can be found)

During the first run, it will compute the COMPARE-16 features using OpenSmile and store them in separate csv files (train_compare_feats.csv and test_compare_feats.csv) 

The trained SVM model (sklearn pipeline of a StandardScaler, and a LinearSVC model) will be saved using pickle

The script will also evaluate the model and generate a prediction file (`test_predicted.csv`) compatible with the submission system

4. After training, the `eval_svm.py` can be used to create predictions on the test set using the model saved in a pickled file

The official pickled SVM baseline model can be found in this repo (see `svm_model.pkl`)

On the test set the model's performance:
| F1 | UAR | Accuracy | MAE (mean absolute error) |
| --- | --- | --- | --- |
| 22.22 | 22.14 | 32.74% | 1.05 |

## End-to-end multi-task wav2vec 2.0 training

To use the multi-task wav2vec2 models, you will need to install https://github.com/aalto-speech/multitask-wav2vec2 (also included in `requirements-mt-w2v2.txt`):
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements-mt-w2v2.txt
```

To train the model, change the ** marked ** parameters and run:
```bash
python train_multitask_wav2vec2_baseline.py \
    --model_name_or_path NbAiLab/nb-wav2vec2-300m-bokmaal \
    --dataset_name **/path_to_the_competition_data** \
    --cache_dir **/cache** \
    --layer_num_for_class 24 \
    --train_split_name train \
    --label_column_name Score \
    --text_column_name Word \
    --output_dir **/mt-w2v2-model** \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --freeze_feature_encoder True \
    --fp16 \
    --learning_rate 2e-4 \
    --attention_mask True \
    --warmup_ratio 0.1 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers "$(nproc)" \
    --logging_strategy steps \
    --logging_steps 100 \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model recall \
    --save_total_limit 2 \
    --seed 0
```
The training takes approx. 1-2 hours on a single GPU (tested with NVIDIA GeForce RTX 2080 Ti and A100)

To generate and store the predictions on the test set, run:
```bash
python eval_multitask_wav2vec2_baseline.py \
    --model_name_or_path **/path_to_model** \
    --tokenizer_name_or_path NbAiLab/nb-wav2vec2-300m-bokmaal \
    --dataset_name **/path_to_the_competition_data** \
    --cache_dir **/cache** \
    --label_column_name Score \
    --text_column_name Word \
    --output_dir **/results_for_codabench** \
    --remove_unused_columns False \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers "$(nproc)"
```

The official multi-task wav2vec 2.0 baseline model is available at [🤗 Hub](TODO). The model's performance on the test set is the following:
|   F1  | UAR   | Accuracy | MAE (mean absolute error) |
| ---   | ---   | ---      | ---                       |
| 38.87 | 37.92 | 55.48    |           0.52            |
