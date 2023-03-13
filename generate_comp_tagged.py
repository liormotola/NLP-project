from inference import perform_inference, read_file_unlabeled,perform_inference_with_roots
from project_evaluate import read_file, compute_metrics
from time import time

def create_labeled_file(german_lines, predicted_english, file_name):
    labeled_file = f"{file_name}_207317744_315046490.labeled"

    with open(labeled_file, "w") as f:
        for ger_sen, en_sen in zip(german_lines, predicted_english):
            f.write("German:\n")
            f.write(f"{ger_sen}\n")
            f.write("English:\n")
            f.write(f"{en_sen}\n")
            f.write("\n")


def generate_tagged_file(unlabeled_file_path,file_name,model_checkpoint):
    german_lines = read_file_unlabeled(unlabeled_file_path)
    s = time()
    generated_english = perform_inference(german_lines,model_checkpoint)
    print(time()-s)


    create_labeled_file(german_lines,generated_english,file_name)

def generate_tagged_file_roots(unlabeled_file_path,file_name,model_checkpoint):
    # german_lines = read_file_unlabeled(unlabeled_file_path)
    s = time()
    german_lines,generated_english = perform_inference_with_roots(unlabeled_file_path,model_checkpoint)
    print(time()-s)


    create_labeled_file(german_lines,generated_english,file_name)

if __name__ == '__main__':

    model_checkpoint = "/home/student/Final Project/Lior/t5-base-translation-from-German-to-English-with_15e5_and_roots/checkpoint-14000"
    #generate val.labeled :
    generate_tagged_file_roots(unlabeled_file_path="data/val.unlabeled",file_name="val_roots",model_checkpoint=model_checkpoint)

    #generate comp.labeled:
    # generate_tagged_file(unlabeled_file_path="data/comp.unlabeled",file_name="comp",model_checkpoint=model_checkpoint)


