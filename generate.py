import pickle
import re
import torch
import argparse

from neural_network import NeuralNetwork
from text_reader import TextReader


def generate_word(reader, prefixes, network):
    inputs = []
    if len(prefixes) < 2 and prefixes[0] == '':
        return reader.random_word()
    for prefix in prefixes:
        vector = reader.word2vector(prefix)
        inputs = [*inputs, *vector]
    data = torch.tensor(inputs)
    output = network(data)
    word = reader.nearest_word(output)
    return word


def generate_text(len_text, init_text, reader, network):
    init_words = re.findall('\w+', init_text.lower())
    words = [*init_words]
    if len(words) < 2:
        words = words + ['']

    while len(words) < len_text:
        prefixes = words[-2:]
        word = generate_word(reader, prefixes, network)
        words.append(word)
        print("> {}".format(word))

    print("Generated text: {}".format(' '.join(words)))


parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--prefix', type=str, default="")
parser.add_argument('--length', type=int, default=3)
args = parser.parse_args()

length = args.length
prefix = args.prefix
reader = TextReader(args.input_dir)
net = NeuralNetwork().to('cpu')
file_name = args.model
with open(file_name, 'rb') as file_model:
    model_params = pickle.load(file_model)
    net.load_state_dict(model_params)
    file_model.close()
    print("Trained model has been deserialized from '{}'".format(file_name))
print(net)

generate_text(length, prefix, reader, net)
