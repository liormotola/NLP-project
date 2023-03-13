import torch.cuda
import pandas as pd
from processing import create_train_df, create_translation_df_from_list , postprocess_text
from project_evaluate import read_file
import project_evaluate
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
import numpy as np
import evaluate
from try_use_roots import create_data_with_roots, create_translation_df_with_roots

tokenizer = AutoTokenizer.from_pretrained("t5-base")

def compute_metrics(eval_preds):

    metric = evaluate.load("sacrebleu")
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    result = {k: round(v, 4) for k, v in result.items()}
    file = open("15e5_roots.txt", "a")
    print(result,file=file)
    print("\n",file=file)
    file.close()
    return result


def preprocess_function(samples):
    """
    This function performs preprocessing to data before sending it to the model. This preprocessing includes
    adding a prefix of the task
    :param samples:
    :return:
    """
    prefix = "translate German to English "

    max_input_length = 256
    max_target_length = 256
    source_lang = "de"
    target_lang = "en"

    inputs = [prefix + sample[source_lang] for sample in samples["translation"]]
    targets = [sample[target_lang] for sample in samples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,padding=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True,padding=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train(train_dataset, test_dataset, batch_size,num_epochs):
    """
    This method performs the training of the model, using hugging face trainer. It trains a t5 model on a given data.
    The trained model is saved to the location saved in output_dir variable in the first line.
    :param train_dataset: train dataset object
    :param test_dataset: test dataset
    :param batch_size: batch size
    :param num_epochs: number of epochs

    """
    output_dir = "t5-base-translation-from-German-to-English-with_15e5_and_roots"
    args = Seq2SeqTrainingArguments(
        output_dir,
        evaluation_strategy = "epoch",
        learning_rate=15e-5,
        generation_max_length = 200,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        greater_is_better = True,
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



def main():
    train_df = create_data_with_roots("data/train.labeled")
    test_df = create_data_with_roots("data/val.labeled")

    # train_df = pd.read_csv("train_data_new.csv")
    # test_df = pd.read_csv("test_data_new.csv")
    # train_text
    # with open("data/train.labeled", "r", encoding='utf8') as train_file:
    #     train_text = train_file.read()
    # train_df = create_train_df()
    # train_file_en, train_file_de = project_evaluate.read_file("data/train.labeled")
    # val_en,val_de = project_evaluate.read_file("data/val.labeled")
    #
    # train_translation_df = create_translation_df(train_df)
    # val_translation_df = create_translation_df(test_df)
    train_translation_df = create_translation_df_with_roots(train_df)
    val_translation_df = create_translation_df_with_roots(test_df)
    # train_translation_df = create_translation_df_from_list(german = train_file_de,english=train_file_en)
    # val_translation_df = create_translation_df_from_list(german=val_de,english=val_en)

    train_dataset = Dataset.from_pandas(train_translation_df)
    validation_dataset = Dataset.from_pandas(val_translation_df)

    initial_datasets = DatasetDict()
    initial_datasets['train'] = train_dataset
    initial_datasets['validation'] = validation_dataset

    tokenized_datasets = initial_datasets.map(preprocess_function, batched=True)
    train(tokenized_datasets['train'],tokenized_datasets['validation'],5,10)


if __name__ == '__main__':

    main()


