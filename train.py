import argparse
import pickle
import torch
import text_reader

from torch.autograd import Variable
from torch.utils.data import DataLoader
from neural_network import NeuralNetwork


def load_data(reader):
    """print("all words: {}".format(
        len(reader.words_emb))
    )"""
    train_data = []
    words_packet = reader.next_packet()
    while words_packet:
        prefix = words_packet[:-1]
        predict = words_packet[-1]
        X = torch.tensor([*prefix[0], *prefix[1]])
        y = torch.tensor(predict)
        train_data.append((X, y))
        words_packet = reader.next_packet()
    return DataLoader(train_data)


def train_network(network, optimizer, criterion, epochs):
    for epoch in range(epochs):
        for batch_idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            data = Variable(X, requires_grad=True)
            target = Variable(y)

            net_out = network(data)
            loss = criterion(net_out[0], target[0])
            loss.backward()
            optimizer.step()

            """print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {}'.format(
                epoch + 1, batch_idx * len(X), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))"""


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--input-dir', type=str)
args = parser.parse_args()
input_dir = args.input_dir
reader = text_reader.TextReader(input_dir)
train_loader = load_data(reader)
"""for X, y in train_loader:
    print(f"Value of X : {X.shape}")
    print(f"Value of y: {y.shape}")"""

net = NeuralNetwork().to('cpu')
print(net)

epochs = 1
learning_rate = 0.0001
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
train_network(net, optimizer, criterion, epochs)

file_name = args.model
with open(file_name, 'wb') as file_model:
    pickle.dump(net.state_dict(), file_model)
    file_model.close()
    print("Trained model has been serialized in '{}'".format(file_name))
