import numpy as np
import re
from pathlib import Path

from model import GNNModel
from losses import BinaryCrossEntropy
from optim import Adam
import utils


def train():
    graphs, labels = load_data("datasets/train")
    train_inputs, train_targets, val_inputs, val_targets = utils.split_train_val(graphs, labels, val_rate=0.3)

    model = GNNModel(8)
    loss_func = BinaryCrossEntropy()
    optimizer = Adam()
    batch_generator = utils.BatchGenerator(batch_size=32)

    min_loss = 100000
    for epoch in range(50):
        print(f"Epoch{epoch + 1}")
        
        train_losses = []
        for inputs, targets in batch_generator.generator(train_inputs, train_targets):
            train_loss, loss_grad = loss_func(model, inputs, targets, is_grad=True)
            optimizer.update(model, loss_grad)
            
            train_losses.append(train_loss)
        
        train_mean_loss = np.mean(train_losses)
        pred = np.array([model.predict(input_) for input_ in train_inputs]).squeeze()
        train_accuracy = accuracy(pred, train_targets)
        
        val_losses = []
        for inputs, targets in batch_generator.generator(val_inputs, val_targets):
            val_loss, _ = loss_func(model, inputs, targets, is_grad=False)
            val_losses.append(val_loss)
        
        val_mean_loss = np.mean(val_losses)
        pred = np.array([model.predict(input_) for input_ in val_inputs]).squeeze()
        val_accuracy = accuracy(pred, val_targets)
        
        
        if min(min_loss, val_mean_loss) < min_loss:
            min_loss = val_mean_loss
            print(f"Train loss: {train_mean_loss}\tTrain accuracy: {train_accuracy}")
            print(f"Validation loss: {val_mean_loss}\tValidation accuracy: {val_accuracy}")
            print("")


def accuracy(pred, true):
    return (pred == true).mean()


def load_data(train_datasets_path):
    graphs = []
    labels = []
    for graph_path in Path().glob(f'{train_datasets_path}/*_graph.txt'):
        file_number = re.search(r'\d+', str(graph_path)).group(0)
        
        with graph_path.open() as fp_graph:
            mat = [row.strip().split() for row in fp_graph.readlines()[1:]]
            mat = np.array(mat, dtype=np.int)
            graphs.append(mat)
            
        with open(fr'{train_datasets_path}/{file_number}_label.txt', 'r') as fp_label:
            label = int(fp_label.read().strip())
            labels.append(label)
            
    return np.array(graphs), np.array(labels)


if __name__ == "__main__":
    train()