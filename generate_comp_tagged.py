from new_inf import perform_inference, read_file_unlabeled
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


def generate_tagged_file(unlabeled_file_path,file_name, labeled_file_path="", calc_metrics =False):
    german_lines = read_file_unlabeled(unlabeled_file_path)
    s = time()
    generated_english = perform_inference(german_lines)
    print(time()-s)
    if labeled_file_path and calc_metrics:
        true_english, _ = read_file(labeled_file_path)
        res = compute_metrics(generated_english, true_english)
        print(f"Bleu result: {res}")

    create_labeled_file(german_lines,generated_english,file_name)


if __name__ == '__main__':


    #generate val.labeled + calc blue over val:
    generate_tagged_file(unlabeled_file_path="data/val.unlabeled",file_name="val",labeled_file_path="data/val.labeled",calc_metrics=True)

    #generate comp.labeled:
    # generate_tagged_file(unlabeled_file_path="data/comp.unlabeled",file_name="comp")


