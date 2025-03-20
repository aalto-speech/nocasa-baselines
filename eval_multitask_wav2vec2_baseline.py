from datasets import load_dataset
import torch
import logging
import transformers
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForMultiTask,
    HfArgumentParser,
    TrainingArguments
)
import librosa
import os
import sys
from dataclasses import dataclass, field
import torch
from typing import Optional

logger = logging.getLogger(__name__)

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
    eval_file: Optional[str] = field(
        default=None, metadata={"help": "A file containing the validation audio paths and labels."}
    )
    eval_split_name: str = field(
        default="train",
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
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
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
    tokenizer_name_or_path: str = field(
        default="facebook/wav2vec2-base",
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"}
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
    predict_transcript: bool = field(
        default=False,
        metadata={"help": "Whether to predict the ASR transcript with the multi-task model."}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
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
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = load_dataset(
            'csv',
            data_files=f"{data_args.dataset_name}/test.csv",
            split=data_args.eval_split_name,
            cache_dir=model_args.cache_dir
        )

    processor = Wav2Vec2Processor.from_pretrained(
        model_args.tokenizer_name_or_path,
        cache_dir=model_args.cache_dir
        )

    model = Wav2Vec2ForMultiTask.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
        ).to(device)

    def map_to_array(example):
        speech, _ = librosa.load(
            f"{data_args.dataset_name}/test/{example['File Name']}",
            sr=16000,
            mono=True
            )
        example["speech"] = speech
        return example

    def map_to_pred_multitask(batch):
        input_values = processor(batch["speech"], sampling_rate=16000, return_tensors="pt", padding="longest").input_values
        with torch.no_grad():
            logits = model(input_values.to(device)).logits
        if model_args.predict_transcript:
            predicted_ids_ctc = torch.argmax(logits[1], dim=-1)
            transcript = processor.batch_decode(predicted_ids_ctc)
            batch["transcript"] = [transcript]
        predicted_ids = torch.argmax(logits[0], dim=-1)
        batch[data_args.label_column_name] = [int(model.config.id2label[id]) for id in predicted_ids.tolist()]
        return batch

    test_dataset = test_dataset.map(map_to_array)

    result = test_dataset.map(
        map_to_pred_multitask,
        remove_columns=["speech"],
        batched=True,
        batch_size=training_args.eval_batch_size
        )

    result.to_csv(
        f"{training_args.output_dir}/{'_'.join(model_args.model_name_or_path.split('/')[-2:])}.csv",
        encoding='utf-8',
        index=False
        )
    
if __name__ == "__main__":
    main()
