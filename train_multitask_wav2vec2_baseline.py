#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import librosa
import sys
import warnings
from dataclasses import dataclass, field
from random import randint

import torch
from typing import Any, Dict, List, Optional, Union
import evaluate
from sklearn.metrics import recall_score, accuracy_score, f1_score
import numpy as np
from datasets import DatasetDict, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    Wav2Vec2FeatureExtractor,
    AutoProcessor,
    Wav2Vec2Processor,
    AutoTokenizer,
    AutoModelForAudioClassification,
    Wav2Vec2ForMultiTask,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.50.0.dev0")

# require_version("datasets>=1.14.0", "To fix: pip install -r examples/pytorch/audio-classification/requirements.txt")

def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(default=None, metadata={"help": "Name of a dataset"})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A file containing the training audio paths and labels."}
    )
    eval_file: Optional[str] = field(
        default=None, metadata={"help": "A file containing the validation audio paths and labels."}
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to 'validation'"
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"}
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"}
    )
    label_column_name: str = field(
        default="label", metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_length_seconds: float = field(
        default=20,
        metadata={"help": "Audio clips will be randomly cut to this length during training if the value is set."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/wav2vec2-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from the Hub"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    layer_num_for_class: int = field(
        default=24,
        metadata={"help": "The transformer layer number from which embeddings are extracted for the classification head."}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    attention_mask: bool = field(
        default=True, metadata={"help": "Whether to generate an attention mask in the feature extractor."}
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=None, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

    def __post_init__(self):
        if not self.freeze_feature_extractor and self.freeze_feature_encoder:
            warnings.warn(
                "The argument `--freeze_feature_extractor` is deprecated and "
                "will be removed in a future version. Use `--freeze_feature_encoder` "
                "instead. Setting `freeze_feature_encoder==True`.",
                FutureWarning,
            )
        if self.freeze_feature_extractor and not self.freeze_feature_encoder:
            raise ValueError(
                "The argument `--freeze_feature_extractor` is deprecated and "
                "should not be used in combination with `--freeze_feature_encoder`. "
                "Only make use of `--freeze_feature_encoder`."
            )

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        class_labels = [{"label": feature["label"]} for feature in features]
        class_labels = [{k: v if v is not None else -100 for (k,v) in item.items()} for item in class_labels]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        batch["label"] = torch.tensor([item["label"] if item["label"] or item["label"]>=0 else -100 for item in class_labels],dtype=torch.long)#, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        return batch

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setting `return_attention_mask=True` is the way to get a correctly masked mean-pooling over
    # transformer outputs in the classifier, but it doesn't always lead to better accuracy
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        return_attention_mask=model_args.attention_mask,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    processor =  Wav2Vec2Processor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
        )

    # Initialize our dataset and prepare it for the audio classification task.
    raw_datasets = DatasetDict()

    train_dataset = load_dataset(
        'csv',
        data_files=f"{data_args.dataset_name}/train.csv",
        split=data_args.train_split_name,
        cache_dir=model_args.cache_dir
    )

    train_dataset = train_dataset.class_encode_column(data_args.label_column_name)

    if training_args.do_eval:
        train_dataset = train_dataset.train_test_split(
            test_size=0.1,
            stratify_by_column=data_args.label_column_name,
            shuffle=True,
            seed=training_args.seed
            )
        eval_dataset = train_dataset['test']
        train_dataset = train_dataset['train']

    def prepare_example(example):
        example['file'] = f"{data_args.dataset_name}/train/{example['File Name']}"
        example[data_args.audio_column_name], example["sampling_rate"] = librosa.load(example["file"], sr=16000)
        example["duration_in_seconds"] = len(example[data_args.audio_column_name]) / example["sampling_rate"]
        
        # Keep target words as ASR references only if the pronunciation score is 4 or 5 (label 3 or 4)
        example[data_args.text_column_name] = example[data_args.text_column_name] if example[data_args.label_column_name]>=3 else ''

        return example

    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
    
        batch["input_values"] = processor(batch[data_args.audio_column_name], sampling_rate=batch["sampling_rate"][0]).input_values
        
        batch["label"] = batch[data_args.label_column_name]

        with processor.as_target_processor():
            batch["labels"] = processor(batch[data_args.text_column_name]).input_ids
            batch["labels"] = [i if i!=[processor.tokenizer.pad_token_id] else [-100] for i in batch["labels"]]
        return batch

    train_dataset = train_dataset.map(prepare_example)
    train_dataset = train_dataset.map(prepare_dataset, batched=True, batch_size=training_args.per_device_train_batch_size)#, remove_columns=['audio'])

    raw_datasets["train"] = train_dataset

    if training_args.do_eval:
        eval_dataset = eval_dataset.map(prepare_example)
        eval_dataset = eval_dataset.map(prepare_dataset, batched=True, batch_size=training_args.per_device_train_batch_size)#, remove_columns=['audio'])
        raw_datasets["eval"] = eval_dataset

    if data_args.audio_column_name not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--audio_column_name {data_args.audio_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(raw_datasets['train'].column_names)}."
        )

    if data_args.label_column_name not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--label_column_name` to the correct text column - one of "
            f"{', '.join(raw_datasets['train'].column_names)}."
        )

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = sorted(np.unique(np.array(raw_datasets["train"][data_args.label_column_name])))
    labels = [str(label+1) for label in labels]

    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    recall_metric = evaluate.load("recall", cache_dir=model_args.cache_dir)
    wer_metric = evaluate.load("wer", cache_dir=model_args.cache_dir)
    cer_metric = evaluate.load("cer", cache_dir=model_args.cache_dir)

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    def compute_metrics(eval_pred):
        """Computes evaluation metrics on a batch of predictions"""
        pred_logits = eval_pred.predictions[1]
        pred_ids = np.argmax(pred_logits, axis=-1)
        ctc_label_ids = eval_pred.label_ids[1]
        ctc_label_ids[ctc_label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.tokenizer.batch_decode(ctc_label_ids, group_tokens=False)
        pred_str = processor.tokenizer.batch_decode(pred_ids)
    
        classification_label_str = eval_pred.label_ids[0]
        classification_pred_str = np.argmax(eval_pred.predictions[0], axis=1)
    
        if logger.isEnabledFor(logging.DEBUG):
            for prediction,reference,classification_prediction,classification_reference in zip(pred_str, label_str,classification_pred_str, classification_label_str):
                logger.debug(f'reference: "{reference}; {classification_reference}"')
                logger.debug(f'prediction: "{prediction}; {classification_prediction}"')
    
        label_str_filtered = [x for idx,x in enumerate(label_str) if label_str[idx]]
        pred_str_filtered = [x for idx,x in enumerate(pred_str) if label_str[idx]]
        classification_label_str_filtered = [x for idx,x in enumerate(classification_label_str) if classification_label_str[idx]>=0]
        classification_pred_str_filtered = [x for idx,x in enumerate(classification_pred_str) if classification_label_str[idx]>=0]
    
        wer = wer_metric.compute(predictions=pred_str_filtered, references=label_str_filtered)
        cer = cer_metric.compute(predictions=pred_str_filtered, references=label_str_filtered)
        recall = recall_metric.compute(predictions=classification_pred_str_filtered, references=classification_label_str_filtered, average='macro')
      
        return {"wer": wer, "cer": cer, "recall": recall["recall"]}

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        cache_dir=model_args.cache_dir,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        layer_num_for_class=model_args.layer_num_for_class,
        vocab_size=len(processor.tokenizer)
    )

    model = Wav2Vec2ForMultiTask.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir
    )

    # freeze the convolutional waveform encoder
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            raw_datasets["train"] = (
                raw_datasets["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = (
                raw_datasets["eval"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["eval"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "audio-classification",
        "dataset": data_args.dataset_name,
        "tags": ["audio-classification"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

if __name__ == "__main__":
    main()
