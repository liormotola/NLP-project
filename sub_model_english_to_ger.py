import torch.cuda
import pandas as pd
from processing import create_train_df, create_raw_data , postprocess_text
from project_evaluate import read_file
import project_evaluate
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
import numpy as np
import evaluate

tokenizer = AutoTokenizer.from_pretrained("t5-base")

def preprocess_function(examples):
    prefix = "translate English to German: "

    max_input_length = 180
    max_target_length = 180
    source_lang = "en"
    target_lang = "de"

    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_preds):

    metric = evaluate.load("sacrebleu")
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def train(train_dataset, test_dataset, batch_size,num_epochs):
    model_name = "t5-base-translation-from-English-to-German neww - checkpoints"
    args = Seq2SeqTrainingArguments(
        model_name,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        generation_max_length = 180,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        save_strategy= "epoch",
    )

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors ='pt')

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()

if __name__ == '__main__':
    train_df = pd.read_csv("train_data_new.csv")
    test_df = pd.read_csv("test_data_new.csv")
    train_raw_df = create_raw_data(train_df)
    val_raw_df = create_raw_data(test_df)
    train_dataset = Dataset.from_pandas(train_raw_df)
    validation_dataset = Dataset.from_pandas(val_raw_df)

    raw_datasets = DatasetDict()
    raw_datasets['train'] = train_dataset
    raw_datasets['validation'] = validation_dataset
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    train(tokenized_datasets['train'], tokenized_datasets['validation'], 8, 20)