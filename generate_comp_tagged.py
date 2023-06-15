from inference import perform_inference_with_roots


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


def generate_val_tagged_file(model_checkpoint):
    """
    loads model from checkpoint, performs inference over the val.unlabeled file and saves the results to a file
    named "val_207317744_315046490.labeled".
    :param model_checkpoint: path to pretrained model to perform inference
    """

    german_lines,generated_english = perform_inference_with_roots("data/val.unlabeled",model_checkpoint)
    create_labeled_file(german_lines,generated_english,"val")

def generate_comp_tagged_file(model_checkpoint):
    """
    loads model from checkpoint, performs inference over comp.unlabeled file and saves the results to a file
    named "comp_207317744_315046490.labeled".
    :param model_checkpoint: path to pretrained model to perform inference
    """

    german_lines,generated_english = perform_inference_with_roots("data/comp.unlabeled",model_checkpoint)
    create_labeled_file(german_lines,generated_english,"comp")

if __name__ == '__main__':
    #unlabeled files should be in a directory called data

    model_checkpoint = "final_model/"

    #generate val.labeled :
    generate_val_tagged_file(model_checkpoint=model_checkpoint)

    #generate comp.labeled:
    generate_comp_tagged_file(model_checkpoint=model_checkpoint)


