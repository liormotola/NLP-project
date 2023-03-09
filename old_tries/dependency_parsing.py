import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import accuracy_score
import math
from time import time
from transformers import AutoModel, get_linear_schedule_with_warmup
import itertools
from typing import List, Set, Tuple, Dict
import numpy


def decode_mst(energy: numpy.ndarray,
               length: int,
               has_labels: bool = True) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Note: Counter to typical intuition, this function decodes the _maximum_
    spanning tree.

    Decode the optimal MST tree with the Chu-Liu-Edmonds algorithm for
    maximum spanning arborescences on graphs.

    Parameters
    ----------
    energy : ``numpy.ndarray``, required.
        A tensor with shape (num_labels, timesteps, timesteps)
        containing the energy of each edge. If has_labels is ``False``,
        the tensor should have shape (timesteps, timesteps) instead.
    length : ``int``, required.
        The length of this sequence, as the energy may have come
        from a padded batch.
    has_labels : ``bool``, optional, (default = True)
        Whether the graph has labels or not.
    """
    if has_labels and energy.ndim != 3:
        raise ValueError("The dimension of the energy array is not equal to 3.")
    elif not has_labels and energy.ndim != 2:
        raise ValueError("The dimension of the energy array is not equal to 2.")
    input_shape = energy.shape
    max_length = input_shape[-1]

    # Our energy matrix might have been batched -
    # here we clip it to contain only non padded tokens.
    if has_labels:
        energy = energy[:, :length, :length]
        # get best label for each edge.
        label_id_matrix = energy.argmax(axis=0)
        energy = energy.max(axis=0)
    else:
        energy = energy[:length, :length]
        label_id_matrix = None
    # get original score matrix
    original_score_matrix = energy
    # initialize score matrix to original score matrix
    score_matrix = numpy.array(original_score_matrix, copy=True)

    old_input = numpy.zeros([length, length], dtype=numpy.int32)
    old_output = numpy.zeros([length, length], dtype=numpy.int32)
    current_nodes = [True for _ in range(length)]
    representatives: List[Set[int]] = []

    for node1 in range(length):
        original_score_matrix[node1, node1] = 0.0
        score_matrix[node1, node1] = 0.0
        representatives.append({node1})

        for node2 in range(node1 + 1, length):
            old_input[node1, node2] = node1
            old_output[node1, node2] = node2

            old_input[node2, node1] = node2
            old_output[node2, node1] = node1

    final_edges: Dict[int, int] = {}

    # The main algorithm operates inplace.
    chu_liu_edmonds(length, score_matrix, current_nodes,
                    final_edges, old_input, old_output, representatives)

    heads = numpy.zeros([max_length], numpy.int32)
    if has_labels:
        head_type = numpy.ones([max_length], numpy.int32)
    else:
        head_type = None

    for child, parent in final_edges.items():
        heads[child] = parent
        if has_labels:
            head_type[child] = label_id_matrix[parent, child]

    return heads, head_type


def chu_liu_edmonds(length: int,
                    score_matrix: numpy.ndarray,
                    current_nodes: List[bool],
                    final_edges: Dict[int, int],
                    old_input: numpy.ndarray,
                    old_output: numpy.ndarray,
                    representatives: List[Set[int]]):
    """
    Applies the chu-liu-edmonds algorithm recursively
    to a graph with edge weights defined by score_matrix.

    Note that this function operates in place, so variables
    will be modified.

    Parameters
    ----------
    length : ``int``, required.
        The number of nodes.
    score_matrix : ``numpy.ndarray``, required.
        The score matrix representing the scores for pairs
        of nodes.
    current_nodes : ``List[bool]``, required.
        The nodes which are representatives in the graph.
        A representative at it's most basic represents a node,
        but as the algorithm progresses, individual nodes will
        represent collapsed cycles in the graph.
    final_edges: ``Dict[int, int]``, required.
        An empty dictionary which will be populated with the
        nodes which are connected in the maximum spanning tree.
    old_input: ``numpy.ndarray``, required.
    old_output: ``numpy.ndarray``, required.
    representatives : ``List[Set[int]]``, required.
        A list containing the nodes that a particular node
        is representing at this iteration in the graph.

    Returns
    -------
    Nothing - all variables are modified in place.

    """
    # Set the initial graph to be the greedy best one.
    parents = [-1]
    for node1 in range(1, length):
        parents.append(0)
        if current_nodes[node1]:
            max_score = score_matrix[0, node1]
            for node2 in range(1, length):
                if node2 == node1 or not current_nodes[node2]:
                    continue

                new_score = score_matrix[node2, node1]
                if new_score > max_score:
                    max_score = new_score
                    parents[node1] = node2

    # Check if this solution has a cycle.
    has_cycle, cycle = _find_cycle(parents, length, current_nodes)
    # If there are no cycles, find all edges and return.
    if not has_cycle:
        final_edges[0] = -1
        for node in range(1, length):
            if not current_nodes[node]:
                continue

            parent = old_input[parents[node], node]
            child = old_output[parents[node], node]
            final_edges[child] = parent
        return

    # Otherwise, we have a cycle so we need to remove an edge.
    # From here until the recursive call is the contraction stage of the algorithm.
    cycle_weight = 0.0
    # Find the weight of the cycle.
    index = 0
    for node in cycle:
        index += 1
        cycle_weight += score_matrix[parents[node], node]

    # For each node in the graph, find the maximum weight incoming
    # and outgoing edge into the cycle.
    cycle_representative = cycle[0]
    for node in range(length):
        if not current_nodes[node] or node in cycle:
            continue

        in_edge_weight = float("-inf")
        in_edge = -1
        out_edge_weight = float("-inf")
        out_edge = -1

        for node_in_cycle in cycle:
            if score_matrix[node_in_cycle, node] > in_edge_weight:
                in_edge_weight = score_matrix[node_in_cycle, node]
                in_edge = node_in_cycle

            # Add the new edge score to the cycle weight
            # and subtract the edge we're considering removing.
            score = (cycle_weight +
                     score_matrix[node, node_in_cycle] -
                     score_matrix[parents[node_in_cycle], node_in_cycle])

            if score > out_edge_weight:
                out_edge_weight = score
                out_edge = node_in_cycle

        score_matrix[cycle_representative, node] = in_edge_weight
        old_input[cycle_representative, node] = old_input[in_edge, node]
        old_output[cycle_representative, node] = old_output[in_edge, node]

        score_matrix[node, cycle_representative] = out_edge_weight
        old_output[node, cycle_representative] = old_output[node, out_edge]
        old_input[node, cycle_representative] = old_input[node, out_edge]

    # For the next recursive iteration, we want to consider the cycle as a
    # single node. Here we collapse the cycle into the first node in the
    # cycle (first node is arbitrary), set all the other nodes not be
    # considered in the next iteration. We also keep track of which
    # representatives we are considering this iteration because we need
    # them below to check if we're done.
    considered_representatives: List[Set[int]] = []
    for i, node_in_cycle in enumerate(cycle):
        considered_representatives.append(set())
        if i > 0:
            # We need to consider at least one
            # node in the cycle, arbitrarily choose
            # the first.
            current_nodes[node_in_cycle] = False

        for node in representatives[node_in_cycle]:
            considered_representatives[i].add(node)
            if i > 0:
                representatives[cycle_representative].add(node)

    chu_liu_edmonds(length, score_matrix, current_nodes, final_edges, old_input, old_output, representatives)

    # Expansion stage.
    # check each node in cycle, if one of its representatives
    # is a key in the final_edges, it is the one we need.
    found = False
    key_node = -1
    for i, node in enumerate(cycle):
        for cycle_rep in considered_representatives[i]:
            if cycle_rep in final_edges:
                key_node = node
                found = True
                break
        if found:
            break

    previous = parents[key_node]
    while previous != key_node:
        child = old_output[parents[previous], previous]
        parent = old_input[parents[previous], previous]
        final_edges[child] = parent
        previous = parents[previous]


def _find_cycle(parents: List[int],
                length: int,
                current_nodes: List[bool]) -> Tuple[bool, List[int]]:

    added = [False for _ in range(length)]
    added[0] = True
    cycle = set()
    has_cycle = False
    for i in range(1, length):
        if has_cycle:
            break
        # don't redo nodes we've already
        # visited or aren't considering.
        if added[i] or not current_nodes[i]:
            continue
        # Initialize a new possible cycle.
        this_cycle = set()
        this_cycle.add(i)
        added[i] = True
        has_cycle = True
        next_node = i
        while parents[next_node] not in this_cycle:
            next_node = parents[next_node]
            # If we see a node we've already processed,
            # we can stop, because the node we are
            # processing would have been in that cycle.
            if added[next_node]:
                has_cycle = False
                break
            added[next_node] = True
            this_cycle.add(next_node)

        if has_cycle:
            original = next_node
            cycle.add(original)
            next_node = parents[original]
            while next_node != original:
                cycle.add(next_node)
                next_node = parents[next_node]
            break

    return has_cycle, list(cycle)


class DParser(nn.Module):
    def __init__(self, model_name = "bert-base-uncased",hidden_dim = 100):
        super(DParser, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.mlp = nn.Sequential(nn.Linear(self.bert.config.hidden_size * 2, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self,input_ids, attention_mask):
        bert_out = self.bert(input_ids,attention_mask=attention_mask )
        sen = bert_out[0][0][1:-1]

        couples = list(itertools.product(sen, sen))
        couples = torch.stack([torch.cat(tup) for tup in couples])
        scores = self.mlp(couples)
        score_matrix = scores.reshape((len(input_ids[0])-2, len(input_ids[0])-2))
        diag = score_matrix.diag()
        diag = diag.view(-1, diag.shape[0])
        score_matrix.fill_diagonal_(-math.inf)
        score_matrix = torch.cat((diag, score_matrix))

        return score_matrix


def train_model(model, data_sets, optimizer, loss_func, num_epochs: int, batch_size=1, model_name="model.pkl"):
    """
    Trains the given model with the given optimizer and loss function for num_epochs epochs.
    Saves the best model to pickle file called model_name.
    Saves all loss and uas results per epoch for both train and test data
    Prints total training time

    Parameters
    ----------
    model: Model to train
    data_sets: Dict where keys are dataset type and values are the corresponding dataset object.
    optimizer: optimizer to use during training.
    loss_func: loss function to use during training.
    num_epochs: number of training epochs.
    pos2idx: dictionary mapping from pos tags to indices. keys are tags, values are indices.
    batch_size: batch size
    model_name: name of the pkl file which the best performing model will be saved to

    Returns
    -------
    train_loss, train_uas, test_loss, test_uas lists of loss and uas pre epoch for both train and test data.
    """
    out_file = open("dparser_results.txt",'w')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
                    "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False)}
    model.to(device)
    train_loss = []
    test_loss = []
    train_uas = []
    test_uas = []
    best_uas = 0.0
    total_steps = len(data_loaders["train"])*num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    start_time = time()
    for epoch in range(num_epochs):
        print(f"starting epoch {epoch + 1}")
        epoch_start_time = time()
        print(f'Epoch {epoch + 1}/{num_epochs}',file=out_file)
        print('-' * 10,file=out_file)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            labels, preds = [], []
            counter = 0
            batch_loss = torch.tensor(0.0).to(device)
            for sentence in data_loaders[phase]:
                counter += 1
                input_ids = torch.tensor(sentence["input_ids"][:sentence["true_lens"]]).unsqueeze(0).to(device)
                attention_mask = torch.tensor(sentence["attention_mask"][:sentence["true_lens"]]).unsqueeze(0).to(device)
                true_heads = torch.tensor(sentence["labels"][:sentence["true_lens"]]).long().to(device)

                model.zero_grad()
                optimizer.zero_grad()
                if phase == 'train':
                    outputs = model(input_ids, attention_mask)
                    loss = loss_func(outputs.T, true_heads)
                    batch_loss += loss
                    if counter == 5:
                        batch_loss /= counter
                        batch_loss.backward()
                        optimizer.step()
                        scheduler.step()
                        counter = 0
                        batch_loss = torch.tensor(0.0).to(device)
                else:
                    with torch.no_grad():
                        outputs = model(input_ids, attention_mask)
                        loss = loss_func(outputs.T, true_heads)

                root_pad = nn.ConstantPad1d((1, 0), 0)
                outputs = root_pad(outputs)
                pred_heads, _ = decode_mst(outputs.cpu().detach(), sentence["true_lens"].item(), has_labels=False)
                preds += list(pred_heads[1:])

                labels += true_heads.cpu().view(-1).tolist()

                running_loss += loss.item() * batch_size

            epoch_loss = running_loss / len(data_sets[phase])
            epoch_acc = accuracy_score(labels, preds)

            epoch_acc = round(epoch_acc, 5)
            if phase == "train":
                train_loss.append(epoch_loss)
                train_uas.append(epoch_acc)
            else:
                test_loss.append(epoch_loss)
                test_uas.append(epoch_acc)

            if phase.title() == "test":
                print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}',file=out_file)
                print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}')

            else:
                print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}',file=out_file)
                print(f'{phase.title()} Loss: {epoch_loss:.4e} Accuracy: {epoch_acc}')

            if phase == 'test' and epoch_acc > best_uas:
                best_uas = epoch_acc
                with open(model_name, 'wb+') as f:
                    torch.save(model, f)

        print(f"Epoch time: {round((time() - epoch_start_time), 3)} sec",file=out_file)
        print(file=out_file)

    print(f"Total training time: {round((time() - start_time), 3)} sec",file=out_file)
    print(f'Best Validation Accuracy: {best_uas:4f}',file=out_file)
    out_file.close()
    return train_loss, train_uas, test_loss, test_uas

def create_train_df(text):
    """
        This method converts the text into data frame of sentences with their pos tags and their true heads (labels)

    Parameters
    ----------
    text data to convert

    Returns
    -------
    data frame of all data
    """


    sentences = text.split("\n\n")[:-1]

    data = {"token_counter":[],"sentences":[],"POS":[] , "heads":[]}

    for sen in sentences:
        sen = sen.split("\n")
        tokens = []
        sentence = []
        poss = []
        heads = []
        if sen == "":
            continue
        for line in sen:
            try:
                splitted_line = line.split("\t")
                token_counter, word, pos, head = splitted_line[0],splitted_line[1], splitted_line[3],splitted_line[6]
                tokens.append(token_counter)
                sentence.append(word)
                poss.append(pos)
                heads.append(head)
            except:
                print(line)
        data["token_counter"].append(" ".join(tokens))
        data["sentences"].append(" ".join(sentence))
        data["POS"].append(" ".join(poss))
        data["heads"].append(" ".join(heads))

    df = pd.DataFrame(data)
    return df


