# NLP-project

## Description
In this project, we implemented a translation model from German to English.
We were given train, validation, and test datasets of paragraphs in German and their translation in English.<br>
**Our Goal** was to maximize the BLEU performance of the model on the validation and test sets.

## The Model
We used a pre-trained T5-Base Model and made some preprocessing to the data, such that the prompt we sent to the model included the root and 2 of its modifiers of each sentence in English.
## How to run?
### main.py
The file which does all of the heavy lifting is `main.py`.<br>
`main.py` is responsible for loading and preprocessing the data, training the model, and saving the trained model.
It also reports loss and accuracy over train and validation sets.

### generate_comp_tagged.py
This file labels data given a pre-trained model and an unlabeled file.<br>
Meaning after training the model, use this file to produce results over unlabeled data (translating German to English)

### project_evaluate.py
Use the function `calculate_score` of this module to evaluate the model's performance over the train/validation data.
It gets labeled files, where one should contain the true labels and the other the model's predictions.
