import project_evaluate
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
import numpy as np
import evaluate
from processing import create_data_with_roots, create_translation_df_with_roots, create_translation_df_val_with_roots,read_file_unlabeled_with_roots ,postprocess_text


tokenizer = AutoTokenizer.from_pretrained("t5-base")

def compute_metrics(eval_preds):
    """
    Function to compute eval - sacrebleu metric to be used in the training procedure.
    :param eval_preds: tuple of preds and labels to be evaluated
    :return: dictionary containing the bleu results and the mean length of generated sentences
    """

    metric = evaluate.load("sacrebleu")
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_predictions, decoded_labels = postprocess_text(decoded_predictions, decoded_labels)

    result = metric.compute(predictions=decoded_predictions, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    result = {k: round(v, 4) for k, v in result.items()}

    return result


def preprocess_function(samples):
    """
    This function performs preprocessing to the data before sending it to the model. This preprocessing includes
    adding a prefix of the task "translate German to English" ,tokenizing, truncating and adding padding to the data sentences.
    :param samples: dictionary with the key "translation" holding samples where each sample is a dictionary with the keys "de","en"
    holding the data to be preprocessed.
    :return: preprocessed dataset, including original data, tokenized inputs, attention masks and tokenized labels.
    """
    prefix = "translate German to English "

    max_input_length = 320
    max_target_length = 320
    source_lang = "de"
    target_lang = "en"

    inputs = [prefix + sample[source_lang] for sample in samples["translation"]]
    targets = [sample[target_lang] for sample in samples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,padding=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True,padding=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train(train_dataset, test_dataset, batch_size,num_epochs,out_loc):
    """
    This method performs the training of the model, using hugging face trainer. It trains a t5 model on a given data.
    The trained model is saved to the location given in out_loc.
    :param train_dataset: train dataset object
    :param test_dataset: test dataset
    :param batch_size: batch size
    :param num_epochs: number of epochs
    :param out_loc: output location - where to save the trained model.

    """
    output_dir = out_loc
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
    """
    The main function, runs all process from loading data to preprocessing and training the model
    """

    #create initial training data
    train_df = create_data_with_roots("data/train.labeled")
    train_translation_df = create_translation_df_with_roots(train_df)

    # create initial validation data
    file_de, roots, modifiers = read_file_unlabeled_with_roots("data/val.unlabeled")
    file_en, file_de_labeled = project_evaluate.read_file("data/val.labeled")
    # make sure data was created properly
    for sen1, sen2 in zip(file_de, file_de_labeled):
        if sen1.strip() != sen2.strip():
            raise ValueError('Different Sentences')

    val_translation_df = create_translation_df_val_with_roots(file_de, roots, modifiers,file_en)

    # creating dataset objects
    train_dataset = Dataset.from_pandas(train_translation_df)
    validation_dataset = Dataset.from_pandas(val_translation_df)

    initial_datasets = DatasetDict()
    initial_datasets['train'] = train_dataset
    initial_datasets['validation'] = validation_dataset

    #tokenizing the datasets
    tokenized_datasets = initial_datasets.map(preprocess_function, batched=True)

    #training
    out_dir = "t5-base-translation-from-German-to-English-val"
    train(tokenized_datasets['train'],tokenized_datasets['validation'],5,6,out_loc=out_dir)



if __name__ == '__main__':

    main()



