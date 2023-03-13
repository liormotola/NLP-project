from inference import perform_inference, read_file_unlabeled,perform_inference_with_roots
from time import time

def create_labeled_file(german_lines, predicted_english, file_name):
    """
    given a list of sentences in german and a list of corresponding lines in english generates a labeled file.
    The file will be named -"{file_name}_207317744_315046490.labeled"
    :param german_lines: list of german sentences
    :param predicted_english: list of corresponding sentences in english
    :param file_name: name of the file to save data to

    """
    labeled_file = f"{file_name}_207317744_315046490.labeled"

    with open(labeled_file, "w") as f:
        for ger_sen, en_sen in zip(german_lines, predicted_english):
            f.write("German:\n")
            f.write(f"{ger_sen}\n")
            f.write("English:\n")
            f.write(f"{en_sen}\n")
            f.write("\n")


def generate_tagged_file(unlabeled_file_path,file_name,model_checkpoint):
    """
    loads model from checkpoint, performs inference over unlabeled file and save the results to a file
    named "{file_name}_207317744_315046490.labeled".
    :param unlabeled_file_path: file containing the unlabeled data to perform inference over. needs to be in unlabeled format.
    :param file_name: name of the file to save the results to
    :param model_checkpoint: path to pretrained model to perform inference

    """
    german_lines = read_file_unlabeled(unlabeled_file_path)
    s = time()
    generated_english = perform_inference(german_lines,model_checkpoint)
    print(time()-s)

    create_labeled_file(german_lines,generated_english,file_name)

def generate_tagged_file_roots(unlabeled_file_path,file_name,model_checkpoint):
    """
    loads model from checkpoint, performs inference over unlabeled file and save the results to a file
    named "{file_name}_207317744_315046490.labeled".
    :param unlabeled_file_path: file containing the unlabeled data to perform inference over. needs to be in unlabeled format.
    :param file_name: name of the file to save the results to
    :param model_checkpoint: path to pretrained model to perform inference
    """

    s = time()
    german_lines,generated_english = perform_inference_with_roots(unlabeled_file_path,model_checkpoint)
    print(time()-s)


    create_labeled_file(german_lines,generated_english,file_name)

if __name__ == '__main__':

    model_checkpoint = "/home/student/Final Project/Lior/t5-base-translation-from-German-to-English-with_15e5_and_roots/checkpoint-14000"
    #generate val.labeled :
    generate_tagged_file_roots(unlabeled_file_path="data/val.unlabeled",file_name="val_roots2",model_checkpoint=model_checkpoint)

    #generate comp.labeled:
    # generate_tagged_file(unlabeled_file_path="data/comp.unlabeled",file_name="comp",model_checkpoint=model_checkpoint)


